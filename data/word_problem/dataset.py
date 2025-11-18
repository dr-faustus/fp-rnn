"""
Adapted from https://github.com/jopetty/word-problem/blob/8f910f92e1c70455dcd9376f56032dfc55126188/src/main.py
"""

from pathlib import Path
from ordered_set import OrderedSet
import polars as pl
from enum import StrEnum


from functools import partial
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from datasets import concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerFast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F


class SpecialTokens(StrEnum):
    """Special tokens for tokenizer."""

    PAD = "[PAD]"
    BOS = "[BOS]"
    UNK = "[UNK]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    CLS = "[CLS]"
    MASK = "[MASK]"

    @classmethod
    def values(cls):
        """Return a list of the string values of each special token."""
        return list(map(lambda c: c.value, cls))

    @property
    def index(self):
        """Return the index of the token in the vocabulary.

        Used to get the index of the PAD token when directly modifying tensors.
        """
        return SpecialTokens.values().index(self.value)
    

def tokenize(
    example: dict[str, Tensor],
    tokenizer: PreTrainedTokenizerFast,
    supervised: bool,
) -> dict[str, Tensor]:
    """Tokenize inputs."""
    tokenized = tokenizer(
        example["input"],
        return_tensors="pt",
        padding=True,
    )
    tokenized.pop("attention_mask", None)

    # If output is not supervised (e.g., for MLPs) then we only keep the final target
    # value since its sequence classification, not token classification.
    tokenized["labels"] = tokenizer(
        example["target"],
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    if not supervised:
        tokenized["labels"] = tokenized["labels"][:, -1]

    return tokenized


def pad_collate(
    samples: list[dict[str, Tensor]], pad_token_id: int
) -> dict[str, Tensor]:
    """Collate function for DataLoader.

    Performs channel-wise padding of the inputs and targets.
    """
    # Only pad `labels` if len(labels) > 1,
    channels_to_pad = ["input_ids"]
    if samples[0]["labels"].dim() > 0:
        channels_to_pad.append("labels")

    max_lens = {}
    for c in channels_to_pad:
        max_lens[c] = max([s[c].shape[0] for s in samples])

    for s in samples:
        for c in channels_to_pad:
            if max_lens[c] > s[c].shape[0]:
                s[c] = F.pad(s[c], (0, max_lens[c] - s[c].shape[0]), value=pad_token_id)

    collated = {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
    }

    return collated


def get_dataset(
    group: str,
    k: int,
    strict_len: bool,
    train_size: float,
    data_dir: str | Path,
    supervised: bool = True,
    max_samples: int | None = None
) -> dict:
    """Construct dataset."""
    assert train_size > 0 and train_size <= 1, "`train_size` must be in (0,1]"
    if type(data_dir) == str:
        data_dir = Path(data_dir)
    if strict_len:
        assert k > 1, "`k` must be at least 2"
        data_paths = [data_dir /  f"{group}={i}.csv" for i in [2, k]]
        data_paths = list(OrderedSet(data_paths))
        if not data_paths[0].exists():
            raise FileNotFoundError(f"You must have data for {group}={2}.")
    else:
        data_paths = [data_dir / f"{group}={i}.csv" for i in range(2, k + 1)]
        if not data_paths[0].exists():
            raise FileNotFoundError(f"You must have data for {group}=2.")
        data_paths = [p for p in data_paths if p.exists()]
        data_paths = list(OrderedSet(data_paths))

    # All unique tokens can be found by looking at the k=2 inputs. We create a
    # a dictionary mapping each token to its index in the vocabulary and use this
    # to construct the tokenizer.
    unique_tokens = (
        pl.read_csv(data_paths[0])
        .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
        .explode("input")
        .unique()["input"]
        .to_list()
    )
    unique_tokens = {t: int(t) for t in unique_tokens}

    tokenizer_base = Tokenizer(WordLevel())
    tokenizer_base.pre_tokenizer = WhitespaceSplit()
    tokenizer_base.add_tokens(sorted(list(unique_tokens.keys()), key=lambda x: int(x)))
    tokenizer_base.add_special_tokens(SpecialTokens.values())
    tokenizer_base.post_processor = TemplateProcessing(
        single=f"{SpecialTokens.BOS} $A",
        special_tokens=[
            (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS))
        ],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_base,
        bos_token=SpecialTokens.BOS.value,
        unk_token=SpecialTokens.UNK.value,
        eos_token=SpecialTokens.EOS.value,
        sep_token=SpecialTokens.SEP.value,
        cls_token=SpecialTokens.CLS.value,
        mask_token=SpecialTokens.MASK.value,
        pad_token=SpecialTokens.PAD.value,
    )
    tokenizer.padding_side = "right"
    tokenize_map = partial(tokenize, tokenizer=tokenizer, supervised=supervised)

    # Construct dataset
    if len(data_paths) == 1:
        dataset = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )
        if max_samples is not None:
            num_samples = min(len(dataset), max_samples)
            dataset = dataset.select(range(num_samples))
        if train_size < 1:
            dataset = dataset.train_test_split(train_size=train_size)
    else:
        train_data = [
            load_dataset("csv", data_files=str(d_path), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
            for d_path in data_paths[:-1]
        ]
        k_data = (
            load_dataset("csv", data_files=str(data_paths[-1]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )

        if max_samples is not None:
            k_data = k_data.select(range(min(len(k_data), max_samples)))
            train_data = [t.select(range(min(len(t), max_samples))) for t in train_data]

        train_data = concatenate_datasets(train_data)

        if train_size < 1:
            dataset = k_data.train_test_split(train_size=train_size)
            dataset["train"] = concatenate_datasets([dataset["train"], train_data])
        else:
            dataset = concatenate_datasets([train_data, k_data])

    return {
        "dataset": dataset.with_format("torch"),
        "tokenizer": tokenizer,
        "n_vocab": tokenizer_base.get_vocab_size(with_added_tokens=True),
    }

def get_single_dataset(
    group: str,
    k: int,
    train_size: float,
    data_dir: str | Path,
    supervised: bool = True,
    max_samples: int | None = None
) -> dict:
    """Construct dataset."""
    if type(data_dir) == str:
        data_dir = Path(data_dir)

    data_path = data_dir / f"{group}=2.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"You must have data for {group}=2. in '{data_dir}'")

    # All unique tokens can be found by looking at the k=2 inputs. We create a
    # a dictionary mapping each token to its index in the vocabulary and use this
    # to construct the tokenizer.
    unique_tokens = (
        pl.read_csv(data_path)
        .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
        .explode("input")
        .unique()["input"]
        .to_list()
    )
    unique_tokens = {t: int(t) for t in unique_tokens}

    tokenizer_base = Tokenizer(WordLevel())
    tokenizer_base.pre_tokenizer = WhitespaceSplit()
    tokenizer_base.add_tokens(sorted(list(unique_tokens.keys()), key=lambda x: int(x)))
    tokenizer_base.add_special_tokens(SpecialTokens.values())
    tokenizer_base.post_processor = TemplateProcessing(
        single=f"{SpecialTokens.BOS} $A",
        special_tokens=[
            (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS))
        ],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_base,
        bos_token=SpecialTokens.BOS.value,
        unk_token=SpecialTokens.UNK.value,
        eos_token=SpecialTokens.EOS.value,
        sep_token=SpecialTokens.SEP.value,
        cls_token=SpecialTokens.CLS.value,
        mask_token=SpecialTokens.MASK.value,
        pad_token=SpecialTokens.PAD.value,
    )
    tokenizer.padding_side = "right"
    tokenize_map = partial(tokenize, tokenizer=tokenizer, supervised=supervised)

    # Construct dataset

    data_path = data_dir / f"{group}={k}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"You must have data for {group}={k}. in '{data_dir}'")

    dataset = (
        load_dataset("csv", data_files=str(data_path), split="all")
        .remove_columns(["seed"])
        .map(tokenize_map, batched=True)
        .remove_columns(["input", "target", "token_type_ids"])
    )
    
    if max_samples is not None:
        num_samples = min(len(dataset), max_samples)
        dataset = dataset.select(range(num_samples))
    
    if train_size < 1:
        dataset = dataset.train_test_split(train_size=train_size)
    return {
        "dataset": dataset.with_format("torch"),
        "tokenizer": tokenizer,
        "n_vocab": tokenizer_base.get_vocab_size(with_added_tokens=True),
    }


def get_dataloaders(batch_size:int, train_size, datadict:dict, drop_last: bool = False):

    dataset = datadict["dataset"]
    tokenizer = datadict["tokenizer"]
    collate_fn = partial(pad_collate, pad_token_id=tokenizer.pad_token_id)

    if train_size < 1:
        # sampler_train = RandomSampler(dataset["train"], num_samples=len(dataset["train"]) // 4)
        train_dataloader = DataLoader(
            dataset["train"],
            # sampler=sampler_train,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=drop_last, # workaround: sometimes the compiler complains with varying batchsizes
        )
        # sampler_eval = RandomSampler(dataset["test"], num_samples=len(dataset["test"]) // 4)
        eval_dataloader = DataLoader(
            dataset["test"],
            # sampler=sampler_eval,
            # shuffle=False,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=drop_last, # workaround: sometimes the compiler complains with varying batchsizes
        )
    else:
        # sampler = RandomSampler(dataset, num_samples=len(dataset) // 4)
        train_dataloader = DataLoader(
            dataset,
            # sampler=sampler,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=drop_last, # workaround: sometimes the compiler complains with varying batchsizes
        )
        eval_dataloader = DataLoader(
            dataset,
            # sampler=sampler,
            # shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=drop_last, # workaround: sometimes the compiler complains with varying batchsizes
        )

    return train_dataloader, eval_dataloader