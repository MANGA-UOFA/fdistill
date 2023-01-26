import re
#from util import *
from collections import defaultdict
import nltk
import numpy as np
import os, time, subprocess, io, sys, re, argparse
from typing import Callable, Dict, Iterable, List, Tuple, Union
from ngram_score import NISTScore, BLEUScore
from rouge_score import rouge_scorer, scoring

from sentence_splitter import add_newline_to_end_of_each_sentence

def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)

def str2bool(s):
    # to avoid issue like this: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if s.lower() in ['t','true','1','y']:
        return True
    elif s.lower() in ['f','false','0','n']:
        return False
    else:
        raise ValueError

def lmap(f: Callable, x: Iterable): # -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def calc_eval_metrics(pred_lns, tgt_lns):
    #bleu_score = np.mean(lmap(
    #    calc_sent_bleu, 
    #    list(zip(pred_lns, tgt_lns))
    #    ))
    pred_lns = [x.rstrip().replace('<|endoftext|>', '') for x in pred_lns]
    bleu = BLEUScore(max_ngram=4)
    nist = NISTScore(max_ngram=4)
    avg_len = 0.
    for ref_sents, hyp in zip(tgt_lns, pred_lns):
        ref_sents = [nltk.tokenize.word_tokenize(sent.lower().replace('<|endoftext|>', '')) \
                     for sent in ref_sents.rstrip().split('\t')]
        hyp = nltk.tokenize.word_tokenize(hyp.lower().replace('<|endoftext|>', ''))
        nist.append(hyp, ref_sents)
        bleu.append(hyp, ref_sents)
        avg_len += len(hyp)
    bleu_score = bleu.score()
    nist_score = nist.score()
    avg_len = avg_len * 1. / len(pred_lns)
    
    return {'BLEU': bleu_score, 'NIST': nist_score, 'average_len': avg_len}


def calc_eval_metrics_corpus(path_ref, path_hyp):
    refs = open(path_ref, encoding='utf-8').readlines()
    hyps = open(path_hyp, encoding='utf-8').readlines()
    return calc_eval_metrics(hyps, refs)

def calc_sent_bleu(pred, ref):
    return nltk.translate.bleu_score.sentence_bleu(
        [nltk.tokenize.word_tokenize(x.lower().replace('<|endoftext|>', '')) for x in ref.split('\t')],
         nltk.tokenize.word_tokenize(pred.replace('<|endoftext|>', '').lower()),
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
    )

def cal_nist(refs, hyps):
    nist = NISTScore(max_ngram=4)
    for ref_sents, hyp in zip(refs, hyps):
        ref_sents = [nltk.tokenize.word_tokenize(sent.lower().replace('<|endoftext|>', '')) for sent in ref_sents.split('\t')]
        hyp = nltk.tokenize.word_tokenize(hyp)
        nist.append(hyp, ref_sents)
    score = nist.score()        
    return score#nltk.translate.nist_score.corpus_nist(refs, hyps, n=4)

def calc_nltk_nist(path_ref, path_hyp):
    refs = open(path_ref, encoding='utf-8').readlines()
    refs = [[nltk.tokenize.word_tokenize(x.lower())] for x in refs]
    hyps = open(path_hyp, encoding='utf-8').readlines()
    hyps = [nltk.tokenize.word_tokenize(x.lower()) for x in hyps]
    return nltk.translate.nist_score.corpus_nist(refs, hyps, n=4)

def calc_nist_bleu(path_refs, path_hyp, fld_out='temp', n_lines=None):
    # call mteval-v14c.pl
    # ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14c.pl
    # you may need to cpan install XML:Twig Sort:Naturally String:Util 

    makedirs(fld_out)

    if n_lines is None:
        n_lines = len(open(path_refs[0], encoding='utf-8').readlines())    
    # import pdb; pdb.set_trace()
    _write_xml([''], fld_out + '/src.xml', 'src', n_lines=n_lines)
    _write_xml([path_hyp], fld_out + '/hyp.xml', 'hyp')#, n_lines=n_lines)
    _write_xml(path_refs, fld_out + '/ref.xml', 'ref')#, n_lines=n_lines)

    time.sleep(1)
    cmd = [
        'perl','/home/lcc/seqkd/chat/3rdparty/mteval-v14c.pl',
        '-s', '%s/src.xml'%fld_out,
        '-t', '%s/hyp.xml'%fld_out,
        '-r', '%s/ref.xml'%fld_out,
        ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # import pdb; pdb.set_trace()
    output, error = process.communicate()

    lines = output.decode().split('\n')

    try:
        nist = lines[-6].strip('\r').split()[1:5]
        bleu = lines[-4].strip('\r').split()[1:5]
        return [float(x) for x in nist], [float(x) for x in bleu]

    except Exception:
        print('mteval-v14c.pl returns unexpected message')
        print('cmd = '+str(cmd))
        print(output.decode())
        print(error.decode())
        return [-1]*4, [-1]*4

def calc_cum_bleu(path_refs, path_hyp):
    # call multi-bleu.pl
    # https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
    # the 4-gram cum BLEU returned by this one should be very close to calc_nist_bleu
    # however multi-bleu.pl doesn't return cum BLEU of lower rank, so in nlp_metrics we preferr calc_nist_bleu
    # NOTE: this func doesn't support n_lines argument and output is not parsed yet

    process = subprocess.Popen(
            ['perl', '3rdparty/multi-bleu.perl'] + path_refs, 
            stdout=subprocess.PIPE, 
            stdin=subprocess.PIPE
            )
    with open(path_hyp, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        process.stdin.write(line.encode())
    output, error = process.communicate()
    return output.decode()


def calc_meteor(path_refs, path_hyp, fld_out='temp', n_lines=None, pretokenized=True):
    # Call METEOR code.
    # http://www.cs.cmu.edu/~alavie/METEOR/index.html

    makedirs(fld_out)
    path_merged_refs = fld_out + '/refs_merged.txt'
    _write_merged_refs(path_refs, path_merged_refs)
    cmd = [
            'java', '-Xmx1g',    # heapsize of 1G to avoid OutOfMemoryError
            '-jar', '3rdparty/meteor-1.5/meteor-1.5.jar', 
            path_hyp, path_merged_refs, 
            '-r', '%i'%len(path_refs),     # refCount 
            '-l', 'en', '-norm'     # also supports language: cz de es fr ar
            ]
    print(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    for line in output.decode().split('\n'):
        if "Final score:" in line:
            return float(line.split()[-1])

    print('meteor-1.5.jar returns unexpected message')
    print("cmd = " + " ".join(cmd))
    print(output.decode())
    print(error.decode())
    return -1 


def calc_len(path, n_lines):
    l = []
    for line in open(path, encoding='utf8'):
        l.append(len(line.strip('\n').split()))
        if len(l) == n_lines:
            break
    return np.mean(l)


def nlp_metrics(path_refs, path_hyp, fld_out='temp',  n_lines=None):
    nist, bleu = calc_nist_bleu(path_refs, path_hyp, fld_out, n_lines)
    meteor = calc_meteor(path_refs, path_hyp, fld_out, n_lines)
    avg_len = calc_len(path_hyp, n_lines)
    return nist, bleu, meteor, avg_len


def _write_merged_refs(paths_in, path_out, n_lines=None):
    # prepare merged ref file for meteor-1.5.jar (calc_meteor)
    # lines[i][j] is the ref from i-th ref set for the j-th query

    lines = []
    for path_in in paths_in:
        lines.append([line.strip('\n') for line in open(path_in, encoding='utf-8')])

    with open(path_out, 'w', encoding='utf-8') as f:
        for j in range(len(lines[0])):
            for i in range(len(paths_in)):
                f.write(str(lines[i][j]) + "\n")



def _write_xml(paths_in, path_out, role, n_lines=None):
    # prepare .xml files for mteval-v14c.pl (calc_nist_bleu)
    # role = 'src', 'hyp' or 'ref'

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE mteval SYSTEM "">',
        '<!-- generated by https://github.com/golsun/NLP-tools -->',
        '<!-- from: %s -->'%paths_in,
        '<!-- as inputs for ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14c.pl -->',
        '<mteval>',
        ]

    for i_in, path_in in enumerate(paths_in):

        # header ----

        if role == 'src':
            lines.append('<srcset setid="unnamed" srclang="src">')
            set_ending = '</srcset>'
        elif role == 'hyp':
            lines.append('<tstset setid="unnamed" srclang="src" trglang="tgt" sysid="unnamed">')
            set_ending = '</tstset>'
        elif role == 'ref':
            lines.append('<refset setid="unnamed" srclang="src" trglang="tgt" refid="ref%i">'%i_in)
            set_ending = '</refset>'
        
        lines.append('<doc docid="unnamed" genre="unnamed">')

        # body -----

        if role == 'src':
            body = ['__src__'] * n_lines
        else:
            with open(path_in, 'r', encoding='utf-8') as f:
                body = f.readlines()
            if n_lines is not None:
                body = body[:n_lines]
        #for i in range(len(body)):
        i = 0
        for b in body:
            line = b.strip('\n')
            line = line.replace('&',' ').replace('<',' ')        # remove illegal xml char
            # if len(line) > 0:
            lines.append('<p><seg id="%i"> %s </seg></p>'%(i + 1, line))
            i += 1

        # ending -----

        lines.append('</doc>')
        if role == 'src':
            lines.append('</srcset>')
        elif role == 'hyp':
            lines.append('</tstset>')
        elif role == 'ref':
            lines.append('</refset>')

    lines.append('</mteval>')
    with open(path_out, 'w', encoding='utf-8') as f:
        f.write(str('\n'.join(lines)))

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)