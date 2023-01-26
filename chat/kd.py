#!/usr/bin/env python

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from finetune import SummarizationModule, TranslationModule
from finetune import main as ft_main
from make_student import create_student_by_copying_alternating_layers, get_layers_to_supervise
from transformers import AutoModelForCausalLM, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils import calculate_bleu, check_output_dir, \
                    freeze_params, label_smoothed_nll_loss, use_task_specific_params, make_new_batch_data
from distillation import SummarizationDistiller

# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import generic_train  # noqa

class ChatDistiller(SummarizationDistiller):
    loss_names = ["loss"]

    def __init__(self, hparams):
        super().__init__(hparams)
        self.sample_beams = hparams.sample_beams
        self.do_sample = hparams.do_sample
        self.top_k = hparams.top_k
        self.top_p = hparams.top_p
        self.sample_temperature = hparams.sample_temperature
        self.max_sample_length = hparams.max_sample_length
        self.min_sample_length = hparams.min_sample_length
        self.num_beam_groups = hparams.num_beam_groups
        self.diversity_penalty = hparams.diversity_penalty
        self.beta = hparams.beta
        MAP_KD_METHODS_TO_STEP_FUNC = {
            'seqkd': self._fast_seqkd_step,
            'engine': self._engine_step,
            'kl': self._fast_kl_step,
            'kl_sample': self._kl_step,
            'rkl': self._rkl_step,
            'tvd_symm': self._tvd_symm_step,
            'js': self._js_step,
        }
        self._train_step = MAP_KD_METHODS_TO_STEP_FUNC[hparams.kd_method]

    def calc_ce_loss(self, mask, s_logits, t_logits):
        """Copy pasted from distillbert (transformers/examples/distillation/)"""
        # mask has False at padding_idx
        sel_mask = mask[:, :, None].expand_as(s_logits).bool()
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1), # bottom
                F.softmax(t_logits_slct / self.temperature, dim=-1), # up 
            )
            * (self.temperature) ** 2
        )
        return loss_ce

    def calc_js_loss(self, mask, s_logits, t_logits, s_wght=0.5, t_wght=0.5):
        """Copy pasted from distillbert (transformers/examples/distillation/)"""
        # mask has False at padding_idx
        sel_mask = mask[:, :, None].expand_as(s_logits)
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        q_prob = t_wght*F.softmax(t_logits_slct / self.temperature, dim=-1) \
                + s_wght*F.softmax(s_logits_slct / self.temperature, dim=-1)
        loss_ce = (
            self.ce_loss_fct(
                torch.log(q_prob+1e-10), # bottom
                F.softmax(s_logits_slct / self.temperature, dim=-1) # up
            )
            * (self.temperature) ** 2
        )
        return loss_ce

    def calc_tvd_loss(self, mask, s_logits, t_logits):
        s_logits = F.softmax(s_logits, dim=-1)
        t_logits = F.softmax(t_logits, dim=-1)
        sel_mask = mask[:, :, None].expand_as(s_logits)
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_tvd = (0.5 * torch.abs(s_logits_slct-t_logits_slct)).sum(dim=-1).mean()
        return loss_tvd

    def calc_engine_loss(self, mask, s_logits, t_logits):
        s_logits = F.softmax(s_logits, dim=-1)
        t_logits = F.log_softmax(t_logits, dim=-1)
        sel_mask = mask[:, :, None].expand_as(s_logits)
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_seqkd = s_logits_slct * t_logits_slct
        loss_seqkd = - (loss_seqkd + 1e-10).sum(dim=-1)        
        loss_seqkd = loss_seqkd.mean()
        return loss_seqkd   

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        SummarizationModule.add_model_specific_args(parser, root_dir)
        add_distill_args(parser)
        return parser

    def training_step(self, batch, batch_idx):# -> dict:
        loss = self._train_step(batch)
        logs = {'loss': loss}#{name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum()# + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss, "log": logs}

    def _fast_seqkd_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id

        input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"]
        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        input_ids, lm_logits = input_ids[:, 1:], lm_logits[:, :-1]
        dec_mask = target_mask[:, 1:].bool() 

        loss = ce_loss_fct(lm_logits.contiguous().view(-1, lm_logits.shape[-1]), input_ids.contiguous().view(-1))
        loss = loss * dec_mask.contiguous().view(-1)
        return loss.mean()

    def _engine_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
        student_pred_ids = self._generate(batch)
        new_batch = make_new_batch_data(batch, student_pred_ids, self.tokenizer)
        input_ids, input_mask, target_mask = new_batch["input_ids"], new_batch["attention_mask"], new_batch["target_mask"]
        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"][:, 1:]

        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids, attention_mask=input_mask, use_cache=True)

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        dec_mask = target_mask[:, 1:].bool()
        teacher_logits = teacher_outputs["logits"][:, 1:]
        loss = self.calc_engine_loss(dec_mask, lm_logits, teacher_logits)
        return loss

    def _kl_step(self, batch: dict):# -> torch.tensor:
        pad_token_id = self.tokenizer.pad_token_id        
        
        teacher_pred_ids = self._generate(batch, gen_by_teacher=True)
        new_batch = make_new_batch_data(batch, teacher_pred_ids, self.tokenizer)
        input_ids, input_mask, target_mask = new_batch["input_ids"], new_batch["attention_mask"], new_batch["target_mask"]

        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"][:, 1:]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids, attention_mask=input_mask, use_cache=True)
        teacher_logits = teacher_outputs["logits"][:, 1:]
        dec_mask = target_mask[:, 1:].bool()
        loss = self.calc_ce_loss(dec_mask, lm_logits, teacher_logits)

        return loss

    def _fast_kl_step(self, batch: dict):# -> torch.tensor:
        pad_token_id = self.tokenizer.pad_token_id

        input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"][:, 1:]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        dec_mask = target_mask[:, 1:]#.bool()

        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids, attention_mask=input_mask, use_cache=True)  # since we are not passing labels, never let this default to True)
            teacher_logits = teacher_outputs["logits"][:, 1:]
        loss_ce = self.calc_ce_loss(dec_mask, lm_logits, teacher_logits)

        return loss_ce

    def _generate(self, batch: dict, gen_by_teacher=False):
        model = self.teacher if gen_by_teacher else self.model
        slen = batch["source_ids"].size(1)
        generated_ids = model.generate(
            batch["source_ids"],
            attention_mask=batch["source_attention_mask"],
            num_beams=self.sample_beams,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            temperature=self.sample_temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            min_length=slen+self.min_sample_length,
            max_length=slen+self.max_sample_length)
        generated_ids = generated_ids[:, slen:]
        return generated_ids

    def _tvd_student_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
        student_pred_ids = self._generate(batch)
        new_batch = make_new_batch_data(batch, student_pred_ids, self.tokenizer)
        input_ids, input_mask, target_mask = new_batch["input_ids"], new_batch["attention_mask"], new_batch["target_mask"]

        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"][:, 1:]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        teacher_outputs = self.teacher(input_ids, attention_mask=input_mask, use_cache=True)
        dec_mask = target_mask[:, 1:].bool()
        teacher_logits = teacher_outputs["logits"][:, 1:]
        loss_tvd = self.calc_tvd_loss(dec_mask, lm_logits, teacher_logits)
        return loss_tvd

    def _tvd_teacher_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"][:, 1:]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        teacher_outputs = self.teacher(input_ids, attention_mask=input_mask, use_cache=True)
        dec_mask = target_mask[:, 1:].bool()
        teacher_logits = teacher_outputs["logits"][:, 1:]
        loss_tvd = self.calc_tvd_loss(dec_mask, lm_logits, teacher_logits)
        return loss_tvd

    def _tvd_symm_step(self, batch):
        loss_1 = self._tvd_student_step(batch)
        loss_2 = self._tvd_teacher_step(batch)
        return 0.5 * loss_1 + 0.5 * loss_2

    def _js_student_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
        student_pred_ids = self._generate(batch)
        new_batch = make_new_batch_data(batch, student_pred_ids, self.tokenizer)
        input_ids, input_mask, target_mask = new_batch["input_ids"], new_batch["attention_mask"], new_batch["target_mask"]

        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"][:, 1:]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids, attention_mask=input_mask, use_cache=True)
        teacher_logits = teacher_outputs["logits"][:, 1:]
        dec_mask = target_mask[:, 1:].bool()
        loss_js = self.calc_js_loss(dec_mask, lm_logits, teacher_logits, s_wght=self.beta, t_wght=(1.-self.beta))
        
        return loss_js

    def _js_teacher_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]

        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"][:, 1:]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        teacher_outputs = self.teacher(input_ids, attention_mask=input_mask, use_cache=True)
        dec_mask = target_mask[:, 1:].bool()
        teacher_logits = teacher_outputs["logits"][:, 1:]
        loss_js = self.calc_js_loss(dec_mask, teacher_logits, lm_logits, t_wght=self.beta, s_wght=(1.-self.beta))
        
        return loss_js

    def _js_step(self, batch):
        loss_1 = self._js_student_step(batch)
        loss_2 = self._js_teacher_step(batch)
        return self.beta * loss_1 + (1.- self.beta) * loss_2

    def _rkl_step(self, batch: dict):# -> torch.tensor:
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
        student_pred_ids = self._generate(batch)
        new_batch = make_new_batch_data(batch, student_pred_ids, self.tokenizer)
        input_ids, input_mask, target_mask = new_batch["input_ids"], new_batch["attention_mask"], new_batch["target_mask"]

        student_outputs = self(input_ids, attention_mask=input_mask, use_cache=True)
        lm_logits = student_outputs["logits"][:, 1:]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids, attention_mask=input_mask, use_cache=True)
        teacher_logits = teacher_outputs["logits"][:, 1:]
        dec_mask = target_mask[:, 1:].bool()
        loss = self.calc_ce_loss(dec_mask, teacher_logits, lm_logits)

        return loss




def add_distill_args(parser):
    # NOTE: if --student argument was specified and the teacher and student base models
    # are different, the models still have to have the same tokenizer, specified by
    # --tokenizer_name. So, for example, you can distill from t5_large to t5_small but not
    # from bart to t5. This s because if the tokenizers are different, the output space
    # for the two models is also different and their logits are not comparable.
    parser.add_argument("--teacher", type=str)
    parser.add_argument("--alpha_ce", default=0.8, type=float)
    parser.add_argument("--alpha_mlm", default=0.2, type=float)
    parser.add_argument("--alpha_hid", default=0.0, type=float, required=False)
    parser.add_argument("--disable_monitor", action="store_true", default=False)
    parser.add_argument("--temperature", default=1., type=float)
    parser.add_argument("--student", type=str, required=False)
    parser.add_argument("--student_decoder_layers", default=12, type=int, required=False)
    parser.add_argument("--student_encoder_layers", default=12, type=int, required=False)
    parser.add_argument("--no_teacher", action="store_true", default=False)
    parser.add_argument("--length_penalty", type=float, default=-1)
    parser.add_argument("--supervise_forward", action="store_true", default=False)
    parser.add_argument("--normalize_hidden", action="store_true", default=False)
    parser.add_argument("--kd_method", type=str, 
                            choices=['engine', 'seqkd', 'kl_sample', 
                                     'rkl', 'tvd_symm', 'js'])
    parser.add_argument("--sample_beams", default=2, type=int)
    parser.add_argument("--top_k", default=None, type=int) # default of hgface
    parser.add_argument("--top_p", default=1., type=float) # default of hgface
    parser.add_argument("--beta", default=0.5, type=float) 
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--sample_temperature", default=1., type=float)
    parser.add_argument("--diversity_penalty", default=0., type=float)
    parser.add_argument("--num_beam_groups", default=1, type=int)
    parser.add_argument("--max_sample_length", default=64, type=int)
    parser.add_argument("--min_sample_length", default=3, type=int)


def create_module(args):
    if args.no_teacher:
        module_cls = SummarizationModule
    else:  # DISTILL WITH TEACHER
        module_cls = ChatDistiller
    args.setup_cls: str = module_cls.__name__
    print(f"using module {args.setup_cls}")
    model = module_cls(args)
    return model


def distill_main(args):
    Path(args.output_dir).mkdir(exist_ok=True)
    check_output_dir(args, expected_items=3)

    model = create_module(args)
    return ft_main(args, model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ChatDistiller.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    distill_main(args)
