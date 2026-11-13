"""Run instruction fine-tuning with Accelerate from a fresh process."""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_training.framework.data.post_train.dataloader import (
    InstructionDataset,
    custom_collate_fn,
)
from model_training.framework.data.post_train.instruction_tuning.prompt_style.styles import (
    SFTDatasetPrepare,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Load the notebook dataset, train, and save the resulting model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Instruction JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Model output directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation_steps)

    with args.data.open(encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list) or not data:
        raise ValueError("The input JSON must contain a non-empty list of instructions")
    train_data = data[: int(len(data) * 0.85)]
    if not train_data:
        raise ValueError("The 85% training split is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_style = SFTDatasetPrepare(model_name=args.model, template="model")
    train_data = prompt_style.prepare(train_data)
    dataset = InstructionDataset(train_data, tokenizer, prompt_style)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(
            custom_collate_fn,
            pad_token_id=tokenizer.pad_token_id,
            allowed_max_length=args.max_length,
        ),
    )
    if len(loader) == 0:
        raise ValueError("Training split is smaller than the requested batch size")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    if accelerator.is_main_process:
        logger.info("Training %d examples for %d epochs on %d processes", len(dataset), args.epochs, accelerator.num_processes)
    for epoch in range(args.epochs):
        for inputs, targets in loader:
            with accelerator.accumulate(model):
                logits = model(input_ids=inputs).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
                )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        if accelerator.is_main_process:
            logger.info("Epoch %d complete; last batch loss %.4f", epoch + 1, loss.item())

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        args.output,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output)
        logger.info("Saved model and tokenizer to %s", args.output)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
