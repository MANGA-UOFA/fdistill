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

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import calculate_test_bleu, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params

from collections import defaultdict

logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calc_diversity_list(hyps):
    tokens = [0.0,0.0]
    types = [defaultdict(int), defaultdict(int)]
    for hyp in hyps:
        words = hyp.split()#nltk.tokenize.word_tokenize(hyp)
        for n in range(2):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys()) / max(tokens[0], 1)
    div2 = len(types[1].keys())/ max(tokens[1], 1)
    return div1, div2


def generate_summaries_or_translations(
    examples: List[str],
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    num_beams=5,
    task="translation",
    prefix=None,
    max_length=1024,
    top_k=None,
    do_sample=False,
    **generate_kwargs,
):#-> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    #print(model.model.encoder.embed_tokens.weight.size())
    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""
    torch.manual_seed(42)
    N = 5
    d1_score = 0.
    d2_score = 0.
    for examples_chunk in tqdm(list(chunks(examples, batch_size))):
        #examples_chunk = [prefix + text for text in examples_chunk]
        examples_chunk = ["t" + text[1:] for text in examples_chunk]
        batch = tokenizer(examples_chunk, max_length=max_length, return_tensors="pt", truncation=True, padding="longest").to(device)
        #print(len(tokenizer))
        #print(batch.input_ids.size(), batch.input_ids.max(), batch.input_ids.min())
        summaries = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            num_beams=5,
            do_sample=True,
            **generate_kwargs,
        )
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i in range(0, len(dec), N):
            hypothesis = [x.lstrip() for x in dec[i:i+N]]
            d1, d2 = calc_diversity_list([" ".join(tokenizer.tokenize(x)) for x in hypothesis])
            d1_score += d1
            d2_score += d2
            fout.write("\t".join(hypothesis) + "\n")
            fout.flush()
    print("dist-1 socre:\t", d1_score / len(examples))
    print("dist-2 socre:\t", d2_score / len(examples))
    fout.close()
    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    return 


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
    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--max_input_length", type=int, required=False, default=1024)
    parser.add_argument("--top_k", type=int, required=False, default=None)
    parser.add_argument("--task", type=str, default="translation", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument("--num_beams", type=int, default=5, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dump-args", action="store_true", help="print the custom hparams with the results")
    parser.add_argument("--do_sample", action="store_true", default=False)
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
    if args.n_obs > 0:
        examples = examples[: args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
    runtime_metrics = generate_summaries_or_translations(
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        num_beams=args.num_beams,
        prefix=args.prefix,
        max_length=args.max_input_length,
        top_k=args.top_k,
        do_sample=args.do_sample,
        **parsed_args,
    )

    # Compute scores
    score_fn = calculate_test_bleu# if "translation" in args.task else calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    scores = score_fn(output_lns, reference_lns)

    return 


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate(verbose=True)