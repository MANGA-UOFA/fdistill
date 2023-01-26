#!/usr/bin/env python

import argparse
import datetime
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import make_new_batch_data_lns, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params, batchify, trunc_pred
from metrics import calc_nltk_nist, calc_eval_metrics_corpus, calc_eval_metrics

logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_ppl_step(tokenizer, model, batch):
    pad_token_id = tokenizer.pad_token_id

    input_ids, input_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
    student_outputs = model(input_ids, attention_mask=input_mask, use_cache=True)
    lm_logits = student_outputs["logits"]

    # Same cross entropy vs. label smoothing logic as finetune.py
    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
    #print(lm_logits.size(), response.size())            
    input_ids, lm_logits = input_ids[:, 1:], lm_logits[:, :-1]
    target_mask = target_mask[:, 1:].bool()

    loss = ce_loss_fct(lm_logits.contiguous().view(-1, lm_logits.shape[-1]), input_ids.contiguous().view(-1))
    loss = torch.masked_select(loss, target_mask.contiguous().view(-1))
    return loss.mean()

def generate_summaries_or_translations(
    examples: List[str],
    examples_ref,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    prefix=None,
    max_src_length=512,
    beam_size=5,
    top_p=0.9,
    temperature=1.,
    min_length=3,
    **generate_kwargs,
):# -> Dict:
    fout = Path(out_file).open("w+", encoding="utf-8")
    model_name = str(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  
    tokenizer.padding_side = "left"
    ppl = 0.
    prefix = ""
    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    #if prefix is None:
        
    output_lns = []
    
    src_chunks = list(chunks(examples, batch_size))
    ref_chunks = list(chunks(examples_ref, batch_size))
    for examples_chunk, examples_ref_chunk in tqdm(zip(src_chunks, ref_chunks), total=len(src_chunks)):
        examples_chunk = [prefix + text for text in examples_chunk]
        batch = make_new_batch_data_lns(examples_chunk, examples_ref_chunk, tokenizer)
        with torch.no_grad():
            ppl += eval_ppl_step(tokenizer, model, batch)
        batch = batchify(examples_chunk, tokenizer, 512)
        #tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(device)
        slen = batch["input_ids"].size(1)
        summaries = model.generate(
            batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            num_beams=beam_size,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=slen+64,
            length_penalty=1.,
            min_length=slen+min_length,
            **generate_kwargs,
        )
        summaries = summaries[:, slen:]

        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        for hypothesis in dec:
            hypothesis = trunc_pred(hypothesis.lstrip()).lstrip()
            output_lns.append(hypothesis)
            fout.write(hypothesis + "\n")
            fout.flush()
    fout.close()
    scores = calc_eval_metrics(output_lns, examples_ref)
    scores["ppl"] = ppl.cpu().item() / len(examples)
    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    return scores

def upper_first(str_):
    if len(str_) > 1:
        return str_[0].upper() + str_[1:]
    else:
        return str_[0].upper()

def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def trunc_turns(source_input):
    sep = '<|endoftext|>'
    source_input = sep.join(source_input.split(sep)[-15:])+sep
    return source_input

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
    parser.add_argument("--max_src_length", type=int, default=400, required=False)
    parser.add_argument("--min_length", type=int, default=3, required=False)
    parser.add_argument("--num_beams", type=int, default=5, required=True)
    parser.add_argument("--temperature", type=float, default=1., required=True)
    parser.add_argument("--top_p", type=float, default=0.9, required=False)

    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--multiref", action="store_true")
    parser.add_argument("--use_all_source", action="store_true")
    parser.add_argument("--dump-args", action="store_true", help="print the custom hparams with the results")
    parser.add_argument(
        "--info",
        nargs="?",
        type=str,
        const=datetime_now(),
        help="use in conjunction w/ --dump-args to print with the results whatever other info you'd like, e.g. lang=en-ru. If no value is passed, the current datetime string will be used.",
    )

    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]
    if args.n_obs > 0:
        examples = examples[: args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")

    
    torch.manual_seed(1024)
    scores = generate_summaries_or_translations(
        examples,
        reference_lns,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        max_src_length=args.max_src_length,
        beam_size=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        min_length=args.min_length,
        **parsed_args,
    )

    if args.reference_path is None:
        return {}
    return 


if __name__ == "__main__":
    run_generate(verbose=True)
