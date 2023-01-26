#!/usr/bin/env python

import argparse
import datetime
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List
import re

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import read_webnlg_files, calculate_bleu, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params, batchify
from metrics import calc_nltk_nist, calc_eval_metrics_corpus, calc_eval_metrics

logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def generate_summaries_or_translations(
    data_path: str,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    prefix=None,
    max_src_length=128,
    beam_size=5,
    **generate_kwargs,
):# -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    raw_out_file = out_file+'.raw'
    fout = Path(out_file).open("w+", encoding="utf-8")
    frawout = Path(raw_out_file).open("w+", encoding="utf-8")
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()
    model.temperature = 1.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    #file_dict = read_webnlg_files(data_path, tokenizer)
    src_file = data_path+'.src'
    input_lns = [x.rstrip() for x in open(src_file).readlines()]#list(file_dict.keys())
    ref0_file = data_path+'.ref0'
    ref1_file = data_path+'.ref1'
    ref2_file = data_path+'.ref2'
    refs0 = [x.rstrip() for x in open(ref0_file).readlines()]
    refs1 = [x.rstrip() for x in open(ref1_file).readlines()]
    refs2 = [x.rstrip() for x in open(ref2_file).readlines()]
    reference_lns = list(zip(refs0, refs1, refs2))
    #input_lns, reference_lns = 
    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""
    outputs = []
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""
    #input_lns = [x[0] for x in input_lns]
    print(len(input_lns))
    #assert False
    for examples_chunk in tqdm(list(chunks(input_lns, batch_size))):
        examples_chunk = [prefix + text for text in examples_chunk]
        batch = tokenizer(examples_chunk, max_length=128, return_tensors="pt", truncation=True, padding="longest").to(device)
        input_ids = batch.input_ids
        summaries = model.generate(
            input_ids.cuda(),
            attention_mask=input_ids.ne(tokenizer.pad_token_id).float().cuda(),
            #use_cache=True,
            num_beams=beam_size,
            #return_
            repetition_penalty=1.,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=128,
            length_penalty=1.,
            #bad_words_ids=[[628], [198]] if True else None,
            #min_length=input_ids.size(1)+3,
            **generate_kwargs,
        )
        #summaries = summaries[:, input_ids.size(1)-1:]

        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        #print(dec)
        for hypothesis in dec:
            hypothesis = hypothesis.lstrip().rstrip()
            #fout.write(upper_first(hypothesis.lstrip())+'<|endoftext|>' + "\n")
            frawout.write(hypothesis + '\n')
            frawout.flush()
            new_hyp = ' '.join(re.split('(\W)', hypothesis))
            new_hyp = ' '.join(new_hyp.split()).lower()
            fout.write(new_hyp + "\n")
            fout.flush()
            outputs.append(new_hyp)
        #print(dec[:3])    
    frawout.close()
    fout.close()

    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(input_lns)
    return outputs, reference_lns, dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))

def upper_first(str_):
    return str_[0].upper() + str_[1:]

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
    parser.add_argument("--num_beams", type=int, default=5, required=True)

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
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")
    #examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]

    #if args.n_obs > 0:
    #    examples = examples[: args.n_obs]
        
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")


    #if args.use_all_source:
    #    EXCLUDE_IDS = []

    #examples = [x for i, x in enumerate(examples) if not i+1 in EXCLUDE_IDS]
    
    output_lns, reference_lns, runtime_metrics = generate_summaries_or_translations(
        args.input_path,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        max_src_length=args.max_src_length,
        beam_size=args.num_beams,
        **parsed_args,
    )

    if args.reference_path is None:
        return {}

    # Compute scores
    score_fn = calculate_bleu
    #output_lns = [x.rstrip().replace('<|endoftext|>', '') for x in open(args.save_path).readlines()]
    #reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    #reference_lns = [x for i, x in enumerate(reference_lns) if not i+1 in EXCLUDE_IDS]
    #scores: dict = calculate_bleu(output_lns, reference_lns)
    assert len(output_lns) == len(reference_lns)

    def filter_empty_ref(refs):
        return [x for x in refs if x!='']
    reference_lns = [filter_empty_ref(refs) for refs in reference_lns]
    scores = calc_eval_metrics(output_lns, reference_lns, redup=False)
    print(scores)
    #scores.update(runtime_metrics)

    #if args.dump_args:
    #    scores.update(parsed_args)
    #if args.info:
    #    scores["info"] = args.info

    #if verbose:
    #    print(scores)

    #if args.score_path is not None:
    #    json.dump(scores, open(args.score_path, "w"))

    return scores


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate(verbose=True)
