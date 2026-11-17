"""Run instruction fine-tuning with Accelerate from a fresh process."""

import argparse
import logging
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from model_training.framework.data.post_train.instruction_tuning.interfaces import (
    create_sft_dataloaders,
    load_instruction_records,
    prepare_sft_data,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Load the notebook dataset, train, and save the resulting model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Instruction JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Model output directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, required=True)
    parser.add_argument("--test-fraction", type=float, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation_steps)

    records = load_instruction_records(args.data)
    prepared_data = prepare_sft_data(
        records,
        model_name=args.model,
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
    )
    loaders = create_sft_dataloaders(
        prepared_data,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loaders.train)
    model.train()
    if accelerator.is_main_process:
        logger.info(
            "Training %d examples for %d epochs on %d processes",
            len(prepared_data.train_data),
            args.epochs,
            accelerator.num_processes,
        )
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
        prepared_data.tokenizer.save_pretrained(args.output)
        logger.info("Saved model and tokenizer to %s", args.output)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
