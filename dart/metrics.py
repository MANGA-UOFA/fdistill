import re
#from util import *
from collections import defaultdict
import nltk
import numpy as np
import os, time, subprocess, io, sys, re, argparse
from typing import Callable, Dict, Iterable, List, Tuple, Union
#from ngram_score import NISTScore, BLEUScore
from sacrebleu.metrics import BLEU, CHRF, TER

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


def make_refs_list(mulrefs_list):
    max_nref = max([len(x) for x in mulrefs_list])
    mulrefs_list = [x+['']*(max_nref-len(x)) for x in mulrefs_list]
    return list(zip(*mulrefs_list))

def calc_eval_metrics(pred_lns, tgt_lns, redup=True):
    #bleu_score = np.mean(lmap(
    #    calc_sent_bleu, 
    #    list(zip(pred_lns, tgt_lns))
    #    ))
    pred_lns = [x.rstrip().replace('<|endoftext|>', '') for x in pred_lns]
    #bleu = BLEUScore(max_ngram=4)
    print(pred_lns[:3])
    print(tgt_lns[:3])
    if redup:
        pred_tgt = list(zip(pred_lns, tgt_lns))
        pred_tgt = [tuple([x] + y) for x, y in pred_tgt]
        pred_tgt = list(set(pred_tgt))
        pred_lns = [x[0] for x in pred_tgt]
        tgt_lns = [list(x[1:]) for x in pred_tgt]
    
    bleu_score = calc_nltk_bleu(tgt_lns, pred_lns) #calc_corpus_bleu(tgt_lns, pred_lns)
    tgt_lns = make_refs_list(tgt_lns) # list of multiple refs to multi list of sing ref
    avg_len = 0.
    chrf_score = calc_corpus_chrf(tgt_lns, pred_lns)
    ter_score = calc_corpus_ter(tgt_lns, pred_lns)
    div_score = calc_diversity_list(pred_lns)
    avg_len = avg_len * 1. / len(pred_lns)
    #nist_score = cal_nist(tgt_lns, pred_lns)
    
    return {'BLEU': bleu_score, 
            'Diversity-1': div_score[0], 'Diversity-2': div_score[1],
            'TER': ter_score, 'chrF': chrf_score}
    #return bleu_score, etp_score, div_score

def calc_eval_metrics_corpus(path_ref, path_hyp):
    refs = open(path_ref, encoding='utf-8').readlines()
    #refs = [[nltk.tokenize.word_tokenize(x.lower())] for x in refs]
    hyps = open(path_hyp, encoding='utf-8').readlines()
    #hyps = [nltk.tokenize.word_tokenize(x.lower()) for x in hyps]
    return calc_eval_metrics(hyps, refs)

def calc_sent_bleu(pred, ref):
    return nltk.translate.bleu_score.sentence_bleu(
        [nltk.tokenize.word_tokenize(x.lower().replace('<|endoftext|>', '')) for x in ref.split('\t')],
         nltk.tokenize.word_tokenize(pred.replace('<|endoftext|>', '').lower()),
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
    )

def calc_chrF_score(references, hypothesis, num_refs=1., nworder=2., ncorder=6., beta=2.):
    #logging.info('STARTING TO COMPUTE CHRF++...')
    #print('STARTING TO COMPUTE CHRF++...')
    hyps_tmp, refs_tmp = 'hypothesis_chrF', 'reference_chrF'

    # check for empty lists
    references_, hypothesis_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append(refs_)
            hypothesis_.append(hypothesis[i])

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis_))

    linear_references = []
    for refs in references_:
        linear_references.append('*#'.join(refs[:num_refs]))

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    rtxt = codecs.open(refs_tmp, 'r', 'utf-8')
    htxt = codecs.open(hyps_tmp, 'r', 'utf-8')

    try:
        totalF, averageTotalF, totalPrec, totalRec = computeChrF(rtxt, htxt, nworder, ncorder, beta, None)
    except:
        logging.error('ERROR ON COMPUTING CHRF++.')
        print('ERROR ON COMPUTING CHRF++.')
        totalF, averageTotalF, totalPrec, totalRec = -1, -1, -1, -1
    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    logging.info('FINISHING TO COMPUTE CHRF++...')
    #print('FINISHING TO COMPUTE CHRF++...')
    return totalF, averageTotalF, totalPrec, totalRec

def calc_corpus_chrf(tgt_lns, pred_lns):
    chrf = CHRF()
    score = chrf.corpus_score(pred_lns, tgt_lns)
    return float(score.format(score_only=True, is_json=False))

def calc_corpus_ter(tgt_lns, pred_lns):
    ter = TER()
    score = ter.corpus_score(pred_lns, tgt_lns)
    return float(score.format(score_only=True, is_json=False))

def calc_ter_score(references, hypothesis, num_refs):
    logging.info('STARTING TO COMPUTE TER...')
    print('STARTING TO COMPUTE TER...')
    ter_scores = []
    for hyp, refs in zip(hypothesis, references):
        candidates = []
        for ref in refs[:num_refs]:
            if len(ref) == 0:
                ter_score = 1
            else:
                try:
                    ter_score = pyter.ter(hyp.split(), ref.split())
                except:
                    ter_score = 1
            candidates.append(ter_score)

        ter_scores.append(min(candidates))

    logging.info('FINISHING TO COMPUTE TER...')
    #print('FINISHING TO COMPUTE TER...')
    return sum(ter_scores) / len(ter_scores)

def calc_corpus_bleu(tgt_lns, pred_lns):
    bleu = BLEU()
    score = bleu.corpus_score(pred_lns, tgt_lns)
    return float(score.format(score_only=True, is_json=False))
    #for tgts in tgt_lns:
    #    refs += [nltk.tokenize.word_tokenize(x.lower) for x in tgts]
    #refs = [[nltk.tokenize.word_tokenize(x.lower())] for x in tgt_lns]
    #hyps = [nltk.tokenize.word_tokenize(x.lower()) for x in pred_lns]
    #return nltk.translate.nist_score.corpus_nist(refs, hyps, n=4)

def calc_nltk_bleu(tgt_lns, pred_lns):
    tgt_lns = [list(map(lambda x: nltk.tokenize.word_tokenize(x), x)) for x in tgt_lns]
    pred_lns = [nltk.tokenize.word_tokenize(x) for x in pred_lns]
    score = nltk.translate.bleu_score.corpus_bleu(tgt_lns, pred_lns)
    return score

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

def calc_entropy_list(hyps):
    # based on Yizhe Zhang's code
    etp_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    i = 0
    for hyp in hyps:
        i += 1
        words = nltk.tokenize.word_tokenize(hyp)
        for n in range(4):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            etp_score[n] += - v /total * (np.log(v) - np.log(total))

    return etp_score


def calc_entropy(path_hyp, n_lines=None):
    # based on Yizhe Zhang's code
    etp_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    i = 0
    for line in open(path_hyp, encoding='utf-8'):
        i += 1
        words = line.strip('\n').split()
        for n in range(4):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                counter[n][ngram] += 1
        if i == n_lines:
            break

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            etp_score[n] += - v /total * (np.log(v) - np.log(total))

    return etp_score


def calc_len(path, n_lines):
    l = []
    for line in open(path, encoding='utf8'):
        l.append(len(line.strip('\n').split()))
        if len(l) == n_lines:
            break
    return np.mean(l)


def calc_diversity(path_hyp):
    tokens = [0.0,0.0]
    types = [defaultdict(int),defaultdict(int)]
    for line in open(path_hyp, encoding='utf-8'):
        words = line.strip('\n').split()
        for n in range(2):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys())/tokens[0]
    div2 = len(types[1].keys())/tokens[1]
    return [div1, div2]

def calc_diversity_list(hyps):
    tokens = [0.0,0.0]
    types = [defaultdict(int), defaultdict(int)]
    for hyp in hyps:
        words = nltk.tokenize.word_tokenize(hyp)
        for n in range(2):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys()) / max(tokens[0], 1)
    div2 = len(types[1].keys())/ max(tokens[1], 1)
    return div1, div2

def nlp_metrics(path_refs, path_hyp, fld_out='temp',  n_lines=None):
    nist, bleu = calc_nist_bleu(path_refs, path_hyp, fld_out, n_lines)
    meteor = calc_meteor(path_refs, path_hyp, fld_out, n_lines)
    entropy = calc_entropy(path_hyp, n_lines)
    div = calc_diversity(path_hyp)
    avg_len = calc_len(path_hyp, n_lines)
    return nist, bleu, meteor, entropy, div, avg_len


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