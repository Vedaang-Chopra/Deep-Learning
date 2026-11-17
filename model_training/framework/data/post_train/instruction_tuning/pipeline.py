"""Reusable data preparation for instruction fine-tuning experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from model_training.framework.data.post_train.dataloader import (
    CyclingDataLoader,
    InstructionDataset,
    custom_collate_fn,
)
from model_training.framework.data.post_train.instruction_tuning.prompt_style.styles import (
    SFTDatasetPrepare,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedSFTData:
    """Prepared records, tokenizer, and prompt formatter for one SFT experiment."""

    prompt_style: SFTDatasetPrepare
    tokenizer: PreTrainedTokenizerBase
    train_data: Dataset
    validation_data: Dataset
    test_data: Dataset


@dataclass(frozen=True)
class SFTDataLoaders:
    """Native PyTorch loaders suitable for inspection or ``Accelerator.prepare``."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


def load_instruction_records(data_path: Path) -> list[dict[str, Any]]:
    """Load and validate a non-empty JSON list of instruction records."""
    logger.info("load_instruction_records | path=%s", data_path)
    try:
        with data_path.open(encoding="utf-8") as input_file:
            records = json.load(input_file)
    except (OSError, json.JSONDecodeError):
        logger.exception("load_instruction_records | failed path=%s", data_path)
        raise

    if not isinstance(records, list) or not records:
        raise ValueError("Instruction data must be a non-empty JSON list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every instruction record must be a JSON object")

    logger.info("load_instruction_records | records=%d", len(records))
    return records


def download_instruction_records(
    data_dir: Path,
    file_name: str,
    url: str,
) -> list[dict[str, Any]]:
    """Download instruction records once when absent, then load the local JSON file."""
    from model_training.framework.data.post_train.instruction_tuning.raw_data.simple_example.instruct_data import (
        download_and_load_file,
    )

    data_path = data_dir / file_name
    logger.info("download_instruction_records | path=%s", data_path)
    try:
        download_and_load_file(data_dir=data_dir, file_name=file_name, url=url)
    except Exception:
        logger.exception("download_instruction_records | failed path=%s", data_path)
        raise
    records = load_instruction_records(data_path)
    logger.info("download_instruction_records | records=%d", len(records))
    return records


def prepare_sft_data(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    train_fraction: float,
    test_fraction: float,
) -> PreparedSFTData:
    """Format records and create deterministic train, validation, and test splits."""
    if not records:
        raise ValueError("Cannot prepare an empty instruction dataset")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if train_fraction + test_fraction >= 1:
        raise ValueError("train_fraction + test_fraction must leave data for validation")

    logger.info(
        "prepare_sft_data | records=%d model=%s train_fraction=%.3f test_fraction=%.3f",
        len(records),
        model_name,
        train_fraction,
        test_fraction,
    )
    prompt_style = SFTDatasetPrepare(model_name=model_name, template="model")
    tokenizer = prompt_style.tokenizer
    if tokenizer is None:
        raise ValueError("A model name is required to load the SFT tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer needs either a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    prepared_records = prompt_style.prepare(Dataset.from_list(records))
    train_end = int(len(prepared_records) * train_fraction)
    test_end = train_end + int(len(prepared_records) * test_fraction)
    if train_end == 0 or test_end == train_end or test_end >= len(prepared_records):
        raise ValueError("Dataset is too small for the requested train/validation/test split")

    prepared = PreparedSFTData(
        prompt_style=prompt_style,
        tokenizer=tokenizer,
        train_data=prepared_records.select(range(train_end)),
        validation_data=prepared_records.select(range(test_end, len(prepared_records))),
        test_data=prepared_records.select(range(train_end, test_end)),
    )
    logger.info(
        "prepare_sft_data | train=%d validation=%d test=%d",
        len(prepared.train_data),
        len(prepared.validation_data),
        len(prepared.test_data),
    )
    return prepared


def create_sft_dataloaders(
    prepared_data: PreparedSFTData,
    *,
    batch_size: int,
    max_length: int,
    num_workers: int,
) -> SFTDataLoaders:
    """Tokenize prepared splits and build CPU loaders for single or multi-GPU use."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if max_length < 1:
        raise ValueError("max_length must be at least one")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    logger.info(
        "create_sft_dataloaders | batch_size=%d max_length=%d num_workers=%d",
        batch_size,
        max_length,
        num_workers,
    )
    collate = partial(
        custom_collate_fn,
        pad_token_id=prepared_data.tokenizer.pad_token_id,
        allowed_max_length=max_length,
    )

    def make_loader(dataset: Dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        tokenized_dataset = InstructionDataset(
            ds=dataset,
            tokenizer=prepared_data.tokenizer,
            prompt_style=prepared_data.prompt_style,
        )
        return DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=collate,
        )

    loaders = SFTDataLoaders(
        train=make_loader(prepared_data.train_data, shuffle=True, drop_last=True),
        validation=make_loader(
            prepared_data.validation_data, shuffle=False, drop_last=False
        ),
        test=make_loader(prepared_data.test_data, shuffle=False, drop_last=False),
    )
    if len(loaders.train) == 0:
        raise ValueError("Training split is smaller than the requested batch size")
    logger.info(
        "create_sft_dataloaders | train_batches=%d validation_batches=%d test_batches=%d",
        len(loaders.train),
        len(loaders.validation),
        len(loaders.test),
    )
    return loaders


def as_cycling_dataloader(loader: DataLoader, *, max_length: int) -> CyclingDataLoader:
    """Adapt a native loader for the repository's single-process training loop."""
    logger.info("as_cycling_dataloader | batches=%d max_length=%d", len(loader), max_length)
    cycling_loader = CyclingDataLoader(loader, seq_len=max_length)
    logger.info("as_cycling_dataloader | batch_size=%d", cycling_loader.batch_size)
    return cycling_loader


def summarize_sft_data(prepared_data: PreparedSFTData) -> dict[str, Any]:
    """Return small, printable facts for a notebook inspection cell."""
    first_record = prepared_data.train_data[0]
    summary = {
        "train_examples": len(prepared_data.train_data),
        "validation_examples": len(prepared_data.validation_data),
        "test_examples": len(prepared_data.test_data),
        "pad_token_id": prepared_data.tokenizer.pad_token_id,
        "prepared_record": first_record,
        "rendered_prompt": prepared_data.prompt_style.preview(first_record),
    }
    logger.info(
        "summarize_sft_data | train=%d validation=%d test=%d",
        summary["train_examples"],
        summary["validation_examples"],
        summary["test_examples"],
    )
    return summary
