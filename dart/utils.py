import itertools
import json
import linecache
import math
import os
import pickle
import socket
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence

#import git
import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from sentence_splitter import add_newline_to_end_of_each_sentence
from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer
from transformers.file_utils import cached_property
from transformers.models.bart.modeling_bart import shift_tokens_right
from metrics import calc_sent_bleu

import copy

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable):
    # -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs):
    # -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
    return compute_metrics_fn


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data_file = Path(data_dir).joinpath(type_path + ".json")
        #self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        with open(self.data_file) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []
        full_all_tgt_lst = []

        for example in lines_dict:
            rela_lst = []
            temp_triples = ''
            for i, tripleset in enumerate(example['tripleset']):
                subj, rela, obj = tripleset
                rela = rela.lower()
                rela_lst.append(rela)
                if i > 0:
                    temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)
            
            all_tgt = [sent['text'] for sent in example['annotations']]
            for sent in example['annotations']:
                full_tgt_lst.append(sent['text'])
                full_src_lst.append(temp_triples)
                full_rela_lst.append(rela_lst)
                full_all_tgt_lst.append(all_tgt)

        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)
        assert len(full_all_tgt_lst) == len(full_src_lst) == len(full_tgt_lst)
        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
            edited_sents.append(sent)

        self.input_encoding = tokenizer(full_src_lst, add_special_tokens=True, truncation=True, 
                                    max_length=self.max_source_length, return_tensors='pt', padding=True)

        self.output_encoding = tokenizer(full_tgt_lst, add_special_tokens=True, truncation=True, 
                                    max_length=self.max_target_length, return_tensors='pt', padding=True)

        # split into category words:
        ssl_lst = full_rela_lst

        #self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=self.max_source_length)['input_ids']

        self.src_sent = []
        self.tgt_sent = []
        self.trip_len = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0

        separator = tokenizer(tokenizer.bos_token, add_special_tokens=False)['input_ids'][0]
        '''
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.trip_len.append(sep_idx-1) 
            self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
            self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
            self.labels[i][:sep_idx] = [tokenizer.pad_token_id] * sep_idx
            temp_src_len += sep_idx - 1
            temp_tgt_len += len(elem) - (sep_idx - 1)
            temp_count += 1
        '''
        self.all_tgt_text = full_all_tgt_lst
        #print('tgt_avg: ', temp_tgt_len / temp_count)
        #print('src_avg: ', temp_src_len / temp_count)
        #print('ratios: ', temp_src_len / temp_tgt_len)

        '''
        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        #print(self.src_cat[0])
        print()
        print(self.labels[1])
        print(self.examples[1])
        print(edited_sents[1])
        print(self.src_sent[1])
        print(self.tgt_sent[1])
        #print(self.src_cat[1])
        assert len(self.examples) == len(self.labels) == len(self.src_sent) == len(self.tgt_sent)
        '''
        if False:#os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.data_file)
            self.used_char_len = True
        assert min(self.src_lens) > 0, f"found empty line in {self.data_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        #dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.all_tgt_text)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")

def pad_to_left(token_ids, pad_token_id, max_length):
    if len(token_ids) < max_length:
        token_ids = torch.cat([
            torch.tensor([pad_token_id]*(max_length-len(token_ids))).long(), 
            token_ids])
    return token_ids

class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index):# -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        #print(len(self.examples), len(self.labels), len(self.src_sent), len(self.tgt_sent), len(self.all_tgt_text))
        return self.input_encoding['input_ids'][index],\
                self.input_encoding['attention_mask'][index],\
                self.output_encoding['input_ids'][index],\
                self.output_encoding['attention_mask'][index],\
                self.all_tgt_text[index]
                #torch.tensor(self.src_cat[index], dtype=torch.long),\
                #self.trip_len[index]
                

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else 'do_not_pad',
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )

    def _tensorize_batch(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
        ) -> torch.Tensor:
            # In order to accept both lists of lists and lists of Tensors
            if isinstance(examples[0], (list, tuple)):
                examples = [torch.tensor(e, dtype=torch.long) for e in examples]
            length_of_first = examples[0].size(0)
            are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
            if are_tensors_same_length:
                return torch.stack(examples, dim=0)
            else:
                if self.tokenizer._pad_token is None:
                    raise ValueError(
                        "You are attempting to pad samples but the tokenizer you are using"
                        f" ({self.tokenizer.__class__.__name__}) does not have one."
                    )
                return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def collate_fn(self, batch):# -> Dict[str, torch.Tensor]:
        input_ids, input_mask, label_ids, label_mask, all_tgt_text = zip(*batch)
        input_ids = torch.stack(input_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        label_ids = torch.stack(label_ids, dim=0)
        label_mask = torch.stack(label_mask, dim=0)
        #trip_len = torch.tensor(trip_len)
        #src = self._tensorize_batch(cate)
        #src_attn = (src != self.tokenizer.pad_token_id) # src
        #tgt_attn = (batch != self.tokenizer.pad_token_id)
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, input_mask = trim_batch(input_ids, pad_token_id, attention_mask=input_mask)
        target_ids, target_mask = trim_batch(label_ids, pad_token_id, attention_mask=label_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": target_ids,
            "target_attention_mask": target_mask,
            "all_tgt_raw": all_tgt_text
        }


def batchify(examples, tokenizer, max_length):
    batch_encoding = tokenizer(examples, add_special_tokens=True, truncation=True, 
                                max_length=max_length,
                                is_split_into_words=False)
    input_ids = batch_encoding["input_ids"]
    input_ids = [torch.tensor(inp, dtype=torch.long) for inp in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {"input_ids": input_ids, "attention_mask": input_ids.ne(tokenizer.pad_token_id).float()}

def read_webnlg_files(path, tokenizer):
    file_dict = {}

    with open(path) as f:
        lines_dict = json.load(f)

    full_rela_lst = []
    full_src_lst = []
    # full_tgt_lst = []
    total_count = 0

    for example in lines_dict:
        rela_lst = []
        temp_triples = ''
        for i, tripleset in enumerate(example['tripleset']):
            subj, rela, obj = tripleset
            rela = rela.lower()
            rela_lst.append(rela)
            if i > 0:
                temp_triples += ' | '
            temp_triples += '{} : {} : {}'.format(subj, rela, obj)
        
        for sent in example['annotations']:
            if (temp_triples,tuple(rela_lst)) not in file_dict:
                file_dict[(temp_triples,tuple(rela_lst))] = []
                full_src_lst.append(temp_triples)
                full_rela_lst.append(tuple(rela_lst))
            file_dict[(temp_triples,tuple(rela_lst))].append(sent["text"])

    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)

    return file_dict

class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    try:
        repo = git.Repo(search_parent_directories=True)
        repo_infos = {
            "repo_id": str(repo),
            "repo_sha": str(repo.head.object.hexsha),
            "repo_branch": str(repo.active_branch),
            "hostname": str(socket.gethostname()),
        }
        return repo_infos
    except TypeError:
        return {
            "repo_id": None,
            "repo_sha": None,
            "repo_branch": None,
            "hostname": None,
        }


EVAL_KEYS = ["bleu", "entropy", "div"]


def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=EVAL_KEYS,
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


# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        assert False
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        assert False
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        #freeze_params(model.transformer.wte)
        #freeze_params(model.transformer.wpe)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )
