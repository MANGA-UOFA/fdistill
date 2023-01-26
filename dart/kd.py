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
from transformers import AutoModelForSeq2SeqLM, MBartTokenizer, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils import calculate_bleu, check_output_dir, freeze_params, label_smoothed_nll_loss, use_task_specific_params


# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import generic_train  # noqa

class SummarizationDistiller(SummarizationModule):
    """Supports T5, Bart, Pegasus and other models that inherit from Bart."""

    loss_names = ["loss"]

    def __init__(self, hparams):
        assert Path(hparams.data_dir).exists()
        self.output_dir = Path(hparams.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        save_dir = self.output_dir.joinpath("student")
        self.sample_beams = hparams.sample_beams
        hparams.model_name_or_path = str(save_dir)  # Tell lightning we are training the student
        teacher = AutoModelForSeq2SeqLM.from_pretrained(hparams.teacher).eval()
        use_task_specific_params(teacher, hparams.task)  # We copy good generation parameters to student by default
        if hparams.student is not None:
            student = AutoModelForSeq2SeqLM.from_pretrained(hparams.student)
            use_task_specific_params(student, hparams.task)
            e_layer_ids, d_layer_ids = None, None
        else:
            student, e_layer_ids, d_layer_ids = create_student_by_copying_alternating_layers(
                teacher, e=hparams.student_encoder_layers, d=hparams.student_decoder_layers, save_path=save_dir
            )

        if hparams.length_penalty != -1:
            student.config.length_penalty = hparams.length_penalty
        hparams.tokenizer_name = hparams.teacher  # Use teacher's tokenizer
        super().__init__(hparams, model=student, config=student.config)
        assert (
            student.config.model_type == teacher.config.model_type
        ), f"teacher, student model types should be the same, got {student.config.model_type} != {teacher.config.model_type}"

        if student.config.model_type == "t5":
            student_encoder_layers = len(student.get_encoder().block)
            student_decoder_layers = len(student.get_decoder().block)
            teacher_encoder_layers = len(teacher.get_encoder().block)
            teacher_decoder_layers = len(teacher.get_decoder().block)
        else:
            student_encoder_layers = student.config.encoder_layers
            student_decoder_layers = student.config.decoder_layers
            teacher_encoder_layers = teacher.config.encoder_layers
            teacher_decoder_layers = teacher.config.decoder_layers

        self.different_base_models = not (hparams.student is None or hparams.teacher == hparams.student)
        self.do_calc_hidden_loss = (not self.different_base_models) and hparams.alpha_hid > 0
        self.different_encoder = self.different_base_models or (student_encoder_layers != teacher_encoder_layers)
        # self.different_encoder determines whether we need to run the teacher encoder
        self.teacher = teacher
        freeze_params(self.teacher)

        if not self.different_encoder:  # To save RAM, delete teacher encoder and freeze student encoder.
            try:
                del self.teacher.model.encoder
            except AttributeError:  # T5
                del self.teacher.encoder

        if e_layer_ids is None:
            e_layer_ids = list(range(student_encoder_layers))
        if d_layer_ids is None:
            d_layer_ids = list(range(student_decoder_layers))

        self.e_layer_ids, self.d_layer_ids = e_layer_ids, d_layer_ids  # type: List[int], List[int]

        if self.do_calc_hidden_loss:  # Intermediate supervision: Decide which layers to supervise
            if hparams.supervise_forward:
                self.e_matches = get_layers_to_supervise(
                    n_student=len(self.e_layer_ids), n_teacher=teacher_encoder_layers
                )
                self.d_matches = get_layers_to_supervise(
                    n_student=len(self.d_layer_ids), n_teacher=teacher_decoder_layers
                )
            else:  # student layer should emulate hidden states of the teacher layer it was copied from
                self.e_matches = self.e_layer_ids
                self.d_matches = self.d_layer_ids
        else:
            self.e_matches = None
            self.d_matches = None

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.temperature = hparams.temperature
        self.alpha_mlm = hparams.alpha_mlm
        self.alpha_ce = hparams.alpha_ce
        self.alpha_hid = hparams.alpha_hid
        self.top_k = hparams.top_k
        self.top_p = hparams.top_p
        self.do_sample = hparams.do_sample
        self.beta = hparams.beta
        MAP_KD_METHODS_TO_STEP_FUNC = {
            'seqkd': self._seqkd_step,
            'engine': self._engine_step,
            'kl_sample': self._kl_sample_step,
            'rkl': self._rkl_step,
            'tvd_symm': self._tvd_symm_step,
            'js': self._js_step,
            'js_sample': self._js_sample_step,
        }
        self.kd_method = hparams.kd_method
        self._train_step = MAP_KD_METHODS_TO_STEP_FUNC[hparams.kd_method]
        gc.collect()
        torch.cuda.empty_cache()

    def calc_ce_loss(self, mask, s_logits, t_logits):
        """Copy pasted from distillbert (transformers/examples/distillation/)"""
        # mask has False at padding_idx
        sel_mask = mask[:, :, None].expand_as(s_logits)
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
            * 1. #(self.temperature) ** 2
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
            * 1. #(self.temperature) ** 2
        )
        return loss_ce

    def calc_tvd_loss(self, mask, s_logits, t_logits):
        if 'log' in self.kd_method:
            s_logits = F.log_softmax(s_logits, dim=-1)    
            t_logits = F.log_softmax(t_logits, dim=-1)
        else:
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
        loss_engine = - (s_logits_slct * t_logits_slct).sum(dim=-1).mean()
        return loss_engine   

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        SummarizationModule.add_model_specific_args(parser, root_dir)
        add_distill_args(parser)
        return parser

    def training_step(self, batch, batch_idx) -> dict:
        loss = self._train_step(batch)
        logs = {'loss': loss}#{name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss, "log": logs}

    def _seqkd_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(labels)
        else:
            decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        # noinspection PyCallingNonCallable
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=self.do_calc_hidden_loss,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]
        teacher_outputs = self.teacher(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            use_cache=False,  # since we are not passing labels, never let this default to True
        )
        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        # Same behavior as modeling_bart.py, besides ignoring pad_token_id
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        return loss

    def _engine_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        # if isinstance(self.model, T5ForConditionalGeneration):
        #    decoder_input_ids = self.model._shift_right(labels)
        #else:
        #    decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        # noinspection PyCallingNonCallable
        student_pred_ids = self._generate(batch)
        decoder_input_ids = shift_tokens_right(student_pred_ids, pad_token_id)

        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=src_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=False,
                use_cache=False,  # since we are not passing labels, never let this default to True
            )

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        dec_mask = student_pred_ids.ne(pad_token_id)
        loss = self.calc_engine_loss(dec_mask, lm_logits, teacher_outputs["logits"])
        return loss

    def _kl_step(self, batch: dict) -> torch.tensor:
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(labels)
        else:
            decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        # noinspection PyCallingNonCallable
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        with torch.no_grad():
            teacher_outputs = self.teacher(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            use_cache=False,  # since we are not passing labels, never let this default to True
            )
        dec_mask = decoder_input_ids.ne(pad_token_id)
        loss_ce = self.calc_ce_loss(dec_mask, lm_logits, teacher_outputs["logits"])
        return loss_ce

    def _kl_sample_step(self, batch: dict) -> torch.tensor:
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

        # noinspection PyCallingNonCallable
        teacher_pred_ids = self._generate(batch, gen_by_teacher=True)
        decoder_input_ids = shift_tokens_right(teacher_pred_ids, pad_token_id)

        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )

        lm_logits = student_outputs["logits"]

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=src_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=False,
                use_cache=False,  # since we are not passing labels, never let this default to True
            )

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        dec_mask = teacher_pred_ids.ne(pad_token_id)
        loss = self.calc_ce_loss(dec_mask, lm_logits, teacher_outputs["logits"])
        return loss


    def _generate(self, batch: dict, gen_by_teacher=False):

        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        model = self.teacher if gen_by_teacher else self.model
        generated_ids = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            #decoder_start_token_id=self.decoder_start_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            num_beams=self.sample_beams,
            max_length=self.eval_max_length,
        )
        generated_ids = generated_ids[:, 1:] # remove the first dllm eos token id
        return generated_ids
        
    def _tvd_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        student_pred_ids = self._generate(batch)
        decoder_input_ids = shift_tokens_right(student_pred_ids, pad_token_id)
        # noinspection PyCallingNonCallable
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        teacher_outputs = self.teacher(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            use_cache=False,  # since we are not passing labels, never let this default to True
        )
        dec_mask = student_pred_ids.ne(pad_token_id)
        loss_tvd = self.calc_tvd_loss(dec_mask, lm_logits, teacher_outputs["logits"])
        return loss_tvd

    def _tvd_student_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        student_pred_ids = self._generate(batch)
        decoder_input_ids = shift_tokens_right(student_pred_ids, pad_token_id)
        # noinspection PyCallingNonCallable
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        teacher_outputs = self.teacher(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            use_cache=False,  # since we are not passing labels, never let this default to True
        )
        dec_mask = student_pred_ids.ne(pad_token_id)
        loss_tvd = self.calc_tvd_loss(dec_mask, lm_logits, teacher_outputs["logits"])
        return loss_tvd

    def _tvd_teacher_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        # noinspection PyCallingNonCallable
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=src_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=False,
                use_cache=False,  # since we are not passing labels, never let this default to True
            )
        dec_mask = labels.ne(pad_token_id)
        loss_tvd = self.calc_tvd_loss(dec_mask, teacher_outputs["logits"], lm_logits)
        
        return loss_tvd

    def _tvd_symm_step(self, batch):
        loss_1 = self._tvd_student_step(batch)
        loss_2 = self._tvd_teacher_step(batch)
        return 0.5 * loss_1 + 0.5 * loss_2

    def _js_student_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        student_pred_ids = self._generate(batch)
        decoder_input_ids = shift_tokens_right(student_pred_ids, pad_token_id)
    
        # noinspection PyCallingNonCallable
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        with torch.no_grad():
            teacher_outputs = self.teacher(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            use_cache=False,  # since we are not passing labels, never let this default to True
        )
        dec_mask = student_pred_ids.ne(pad_token_id)
        loss_js = self.calc_js_loss(dec_mask, lm_logits, teacher_outputs["logits"], s_wght=self.beta, t_wght=(1.-self.beta))
        
        return loss_js

    def _js_teacher_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        # noinspection PyCallingNonCallable
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=src_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=False,
                use_cache=False,  # since we are not passing labels, never let this default to True
            )
        dec_mask = labels.ne(pad_token_id)
        loss_js = self.calc_js_loss(dec_mask, teacher_outputs["logits"], lm_logits, s_wght=(1-self.beta), t_wght=self.beta)
        
        return loss_js

    def _js_teacher_sample_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

        # noinspection PyCallingNonCallable
        teacher_pred_ids = self._generate(batch, gen_by_teacher=True)
        decoder_input_ids = shift_tokens_right(teacher_pred_ids, pad_token_id)
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )

        lm_logits = student_outputs["logits"]

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=src_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=False,
                use_cache=False,  # since we are not passing labels, never let this default to True
            )

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        dec_mask = teacher_pred_ids.ne(pad_token_id)
        loss_js = self.calc_js_loss(dec_mask, teacher_outputs["logits"], lm_logits, s_wght=(1-self.beta), t_wght=self.beta)
        
        return loss_js


    def _js_step(self, batch):
        loss_1 = self._js_student_step(batch)
        loss_2 = self._js_teacher_step(batch)
        return self.beta * loss_1 + (1-self.beta) * loss_2

    def _js_sample_step(self, batch):
        loss_1 = self._js_student_step(batch)
        loss_2 = self._js_teacher_sample_step(batch)
        return self.beta * loss_1 + (1-self.beta) * loss_2

    def _rkl_step(self, batch: dict) -> torch.tensor:
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

        # noinspection PyCallingNonCallable
        student_pred_ids = self._generate(batch)
        decoder_input_ids = shift_tokens_right(student_pred_ids, pad_token_id)
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )

        lm_logits = student_outputs["logits"]

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=src_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=False,
                use_cache=False,  # since we are not passing labels, never let this default to True
            )

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        dec_mask = student_pred_ids.ne(pad_token_id)
        loss = self.calc_ce_loss(dec_mask, teacher_outputs["logits"], lm_logits)
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
    parser.add_argument("--student", type=str, required=False)
    parser.add_argument("--student_decoder_layers", default=12, type=int, required=False)
    parser.add_argument("--student_encoder_layers", default=12, type=int, required=False)
    parser.add_argument("--no_teacher", action="store_true", default=False)
    parser.add_argument("--length_penalty", type=float, default=-1)
    parser.add_argument("--supervise_forward", action="store_true", default=False)
    parser.add_argument("--normalize_hidden", action="store_true", default=False)
    parser.add_argument("--kd_method", type=str, 
                            choices=['engine', 'seqkd', 
                                     'kl_sample', 
                                     'rkl', 
                                     'tvd_symm',
                                     'js', 'js_sample'])
    parser.add_argument("--sample_beams", default=1, type=int)
    parser.add_argument("--top_k", default=50, type=int) # default of hgface
    parser.add_argument("--top_p", default=1., type=float) # default of hgface
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", default=1., type=float)
    parser.add_argument("--beta", default=0.5, type=float)

def create_module(args):
    if args.no_teacher:
        module_cls = SummarizationModule
    else:  # DISTILL WITH TEACHER
        module_cls = SummarizationDistiller
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
    parser = SummarizationDistiller.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    distill_main(args)
