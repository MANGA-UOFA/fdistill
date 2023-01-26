#!/usr/bin/env python

import argparse
import datetime
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils import read_webnlg_files, calculate_bleu, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params


logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_nmode(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    return (cumulative_probs < top_p).float().sum(dim=-1).mean(dim=0)

    
def generate_summaries_or_translations(
    examples: List[str],
    references: List[str],
    out_file: str,
    model_name: str,
    student_model_name: str,
    batch_size: int = 4,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    num_beams=5,
    task="translation",
    prefix=None,
    **generate_kwargs,
):#-> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    #fout = Path(out_file).open("w", encoding="utf-8")

    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()
    if student_model_name:
        student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    #print(model.model.encoder.embed_tokens.weight.size())
    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    entropy = []
    eos_lprobs = []
    nmodes = []
    tok_loss = torch.zeros(256).cuda()
    top1_hit = 0.
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""
    loss_seq = []
    for examples_chunk, references_chunk in tqdm(
            zip(
                list(chunks(examples, batch_size)),
                list(chunks(references, batch_size))
                )
            ):
        examples_chunk = [prefix + text for text in examples_chunk]
        batch = tokenizer.prepare_seq2seq_batch(
                    examples_chunk,
                    tgt_texts=references_chunk,
                    max_length=256,
                    max_target_length=256,
                    return_tensors="pt",
                ).data
        tgt_ids = batch["labels"].cuda()
        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")
        loss = 0.
        bsz = batch["input_ids"].size(0)
        with torch.no_grad():
            logits = model(
                batch["input_ids"].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                decoder_input_ids=decoder_input_ids.cuda(),
                output_hidden_states=False,
                use_cache=False,  # since we are not passing labels, never let this default to True
            )["logits"]
        top1_pred = torch.argmax(logits, dim=-1)#[:, :-1]
        top1_hit += (top1_pred.eq(tgt_ids).float() * tgt_ids.ne(tokenizer.pad_token_id).float()).sum()
        _loss = ce_loss_fct(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1)).view(bsz, -1)
        all_batch_loss = _loss.sum(0)
        tok_loss[:all_batch_loss.size(-1)] = tok_loss[:all_batch_loss.size(-1)] + all_batch_loss
        loss_seq += (_loss.sum(1)/tgt_ids.ne(tokenizer.pad_token_id).float().sum(1)).cpu().tolist()
        
    print('Tok-Level LOSS:\t', tok_loss / len(examples))
    print('LOSS:\t', np.mean(loss_seq))
    print("Top-1 hit acc:\t", top1_hit/len(examples))
    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))


def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_generate(verbose=True):
    """

    Takes input text, generates output, and then using reference calculates the BLEU scores.

    The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.

    Args:
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout

    Returns:
        a tuple: ``(scores, params}``
        - ``scores``: a dict of scores data ``{'bleu': 39.6501, 'n_obs': 2000, 'runtime': 186, 'seconds_per_sample': 0.093}``
        - ``params``: a dict of custom params, e.g. ``{'num_beams': 5, 'length_penalty': 0.8}``
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--student_model_name", type=str, default=None, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument("--num_beams", type=int, default=5, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dump-args", action="store_true", help="print the custom hparams with the results")
    parser.add_argument(
        "--info",
        nargs="?",
        type=str,
        const=datetime_now(),
        help="use in conjunction w/ --dump-args to print with the results whatever other info you'd like, e.g. lang=en-ru. If no value is passed, the current datetime string will be used.",
    )
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")


    examples = [x.rstrip() for x in open(args.input_path).readlines()]
    references = [x.rstrip() for x in open(args.reference_path).readlines()]

    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
    runtime_metrics = generate_summaries_or_translations(
        examples,
        references,
        args.save_path,
        args.model_name,
        args.student_model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        num_beams=args.num_beams,
        prefix=args.prefix,
        **parsed_args,
    )

    return scores


if __name__ == "__main__":
    run_generate(verbose=True)
