"""Lab 01 — SFT from scratch (starter).

Stubs only: signatures, contract docstrings, TODO markers, shape comments,
and ``raise NotImplementedError``. Implement the bodies yourself in the
notebook first, then paste back here before exporting to a cluster train.py.

NO-SOLUTIONS RULE reminders while you work:
- Do not compute cross-entropy here as a reference; write it once in your
  notebook and understand every term.
- Mask-building must be *your* arithmetic.
- Optimizer/accumulator bookkeeping is deliberately left empty.

Answer key (open ONLY after your version trains):
    rlhf-book/code/instruction_tuning/train.py      (loop structure)
    rlhf-book/code/instruction_tuning/utils.py      (encode/collate/loss)
    rlhf-book/code/instruction_tuning/configs/sft_olmo2_1b.yaml

Import contract: this module imports cleanly WITHOUT torch installed
(torch is optional and guarded). On Colab/cluster everything works with torch.

Local, CPU-only test run:
    python3 -m pytest tests/ -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --- Optional torch --------------------------------------------------------
# torch may be absent on this machine (tests run numpy-only). Modules that need
# real torch ops should import it lazily inside function bodies via
# `_require_torch()` so this file stays importable everywhere.
try:  # pragma: no cover - exercised on GPU machines
    import torch.nn as nn  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - local Mac / CI path
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    import torch.nn as nn


def _require_torch() -> None:
    """Raise a helpful error if torch is not installed in the current env."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for this lab component. "
            "Install it (Colab/cluster) before running training code."
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100  # label value at positions with no gradient

DEFAULT_SAMPLE_PANEL_PROMPTS: List[str] = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "How does photosynthesis work?",
    "Rewrite this sentence in past tense: 'I go to school.'",
    "Give me two healthy breakfast ideas.",
]


# ---------------------------------------------------------------------------
# Batch container (plumbing only — no ML logic)
# ---------------------------------------------------------------------------


@dataclass
class SFTBatch:
    """Rectangular prompt-masked SFT batch.

    Attributes:
        input_ids:      int64 token ids,            shape (B, T)
        attention_mask: 1 for real tokens, 0 at pads, shape (B, T)
        labels:         targets matching input_ids shifted by callers;
                        IGNORE_INDEX(-100) wherever no gradient flows,
                        shape (B, T)
    """

    input_ids: Any
    attention_mask: Any
    labels: Any

    def to(self, device: Any) -> "SFTBatch":
        """Move all tensors to `device` (returns an equivalent new batch)."""
        raise NotImplementedError(
            "TODO(01.data): move each field to device and return a new SFTBatch."
        )


# ---------------------------------------------------------------------------
# Data pipeline: encode -> mask -> collate -> dataloader
# ---------------------------------------------------------------------------


class SFTDataset:
    """Map-style dataset over pre-encoded rows.

    Each encoded row is expected to look like:
        {"input_ids": List[int] | tensor (T_i,),
         "labels":    same length as input_ids}
    Rows where labels are entirely IGNORE_INDEX carry no signal and should be
    filtered out by the builder, never yielded to the collator.
    """

    def __init__(self, encoded_rows: Sequence[Dict[str, Any]]):
        # Plumbing only: store the pre-encoded rows.
        self.rows = list(encoded_rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Plumbing only: return the stored row.
        return self.rows[idx]


def build_prompt_masked_labels(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_length: int = 2048,
) -> Optional[Tuple[List[int], List[int]]]:
    """Render one conversation and mark which positions get gradient.

    Contract:
      - Render the full conversation (`apply_chat_template`, add_generation_prompt=False).
      - Render the prompt-only prefix (all messages except the last assistant turn,
        add_generation_prompt=True). The generation-prompt tokens belong to the MASKED side.
      - Supervise exactly the assistant completion: from the end of the masked prefix
        through EOS inclusive. Everything else gets IGNORE_INDEX.
      - Truncate to `max_length` tokens BEFORE masking (and skip the row if the
        assistant side ends up fully truncated — return None then).
      - Return None if `messages` doesn't end with role=="assistant" or contains
        no trainable positions after truncation.

    Returns:
        (input_ids, labels) each of length T_full (<= max_length), or None if skipped.

    Shapes:
        input_ids: (T_full,)   labels: (T_full,)
    """
    raise NotImplementedError(
        "TODO(01.data): template twice, locate the prompt boundary, build labels with "
        "IGNORE_INDEX outside [prompt_end, EOS]."
    )


def collate_sft(examples: Sequence[Dict[str, Any]], pad_token_id: int) -> SFTBatch:
    """Pad variable-length encoded rows into one rectangular right-padded batch.

    Contract (pad side RIGHT throughout):
      - Find T_max = max(len(row["input_ids"])) over the batch.
      - input_ids:      append `pad_token_id` up to T_max per row.
      - attention_mask: 1 on real tokens, 0 on pad slots.
      - labels:         append IGNORE_INDEX(-100) on pad slots (pads never supervised).
    The output MUST be rectangular ((B, T_max) for all three fields); ragged rows are a bug.

    Shapes:
        input_ids / attention_mask / labels: (B, T_max) long
    """
    raise NotImplementedError("TODO(01.data): right-pad all three arrays to T_max.")


def make_dataloader(cfg: Any, tokenizer: Any) -> Any:
    """Load + encode + batch the configured SFT split into a DataLoader.

    Contract:
      - cfg drives: dataset_name, dataset_split, max_samples, max_length, batch_size,
        shuffle seed. See configs/01_sft_from_scratch.yaml.
      - Encode each conversation row via build_prompt_masked_labels; drop Nones.
      - Wrap rows in SFTDataset, wire collate_sft(pad_token_id=tokenizer.pad_token_id).
      - Log how many source rows were skipped and why (count is your data-quality signal).
    """
    raise NotImplementedError(
        "TODO(01.data): load_dataset -> encode -> filter -> Dataset(DataLoader(collate))."
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(model: Any, batch: SFTBatch) -> Any:
    """Masked causal-LM cross-entropy (a scalar).

    Contract:
      - Forward `model(input_ids, attention_mask, use_cache=False)` once.
      - Shift: predict token t+1 from logits t, i.e. logits[:, :-1] vs labels[:, 1:].
      - Mean CE over ALL positions in the flattened (B*(T-1), V) x (B*(T-1)) pair,
        using ignore_index=IGNORE_INDEX so masked positions contribute nothing.
        (Cross_entropy's ignore_index does the reduction math for you; the point of
        this lab is understanding WHY that equals mean-over-unmasked.)
      - Guard against an all-masked batch (return-safe behavior is yours to choose,
        but document it).

    Shapes:
        logits: (B, T, V) -> shift -> (B, T-1, V) -> flatten -> (B*(T-1), V) vs (B*(T-1),)
        loss: scalar
    """
    raise NotImplementedError(
        "TODO(01.loss): forward once, shift logits vs labels, CE with ignore_index=-100."
    )


# ---------------------------------------------------------------------------
# LoRA-from-scratch checkpoint (the graded from-scratch piece)
# ---------------------------------------------------------------------------


class LoRALinear:
    """A frozen linear layer wrapped with trainable low-rank adapters A·B.

    Contract you implement:
      - Wrap an existing nn.Linear(in, out) WITHOUT copying/replacing its weight:
        base weight W becomes frozen; add two fresh parameters
        A: (r, in_features)  and  B: (out_features, r).
      - forward(x) = base(x) + scaling * B(A(dropout(x))),
        where scaling = alpha / r. Shapes flow:
            x            : (N, in)
            A(x)         : (N, r)
            B(A(x))      : (N, out)
            output       : (N, out)   # rank-r UPDATE on top of frozen Wx
      - Initialization is not free to choose arbitrarily: A ~ Gaussian,
        B = zeros, so training starts exactly at the frozen model.
        Explain WHY both-zero init fails in notebook markdown (§8 Q3).
    """

    def __init__(self, base: Any, r: int, alpha: int) -> None:
        raise NotImplementedError(
            "TODO(01.lora): store frozen base, create A/B parameters (correct shapes), set alpha/r."
        )

    def forward(self, x: Any) -> Any:
        """Return base(x) + (alpha/r) * B(A(x)). Never touch W's values."""
        raise NotImplementedError(
            "TODO(01.lora): two matmuls through A then B, scaled by alpha/r, added to base output."
        )

    def merge(self) -> Any:
        """Fold W + (alpha/r)*B@A into a single equivalent Linear (rank-r weight update).

        Only valid for inference/eval use afterwards. Return the merged module.
        """
        raise NotImplementedError("TODO(01.lora): materialize W += (alpha/r)*B@A once.")


def apply_lora(model: Any, target_modules: Sequence[str], r: int, alpha: int) -> Any:
    """Wrap every named nn.Linear submodule with LoRALinear and freeze originals.

    Contract:
      - Iterate model.named_modules(); wrap any module whose attr-name matches
        target_modules (e.g. ("q_proj", "v_proj")) IN PLACE.
      - Freeze original weights (requires_grad=False); only A/B stay trainable.
      - Return the model (mutated is fine).
    """
    raise NotImplementedError(
        "TODO(01.lora): swap targeted Linears for LoRALinear wrappers, freeze base weights."
    )


def pack_sequences(
    examples: Sequence[Dict[str, Any]],
    pack_length: int,
) -> List[Dict[str, Any]]:
    """OPTIONAL EXERCISE — greedy naive packing into fixed-length bins.

    Contract (deliberately NAIVE — cross-contamination risk is part of the lesson):
      - Greedily concatenate encoded examples until adding the next would exceed
        pack_length, then close the bin and start a new one (first-fit-first, no
        block-diagonal attention, no document separators).
      - attention_mask stays 1 across whole packed sequence (this is what makes it risky).
      - position_ids restart at 0 at every example boundary (so a causal model can at
        least see relative offsets reset; whether that rescues safety is discussed in §8 Q4).
      - labels concatenate unchanged, pads only filled by bin closure IGNORE_INDEX.
      - Drop nothing: trailing examples keep their own (shorter) final bin.

    Returns bins shaped like single examples but with extra keys position_ids, seq_lens:
        {"input_ids","labels","attention_mask","position_ids","seq_lens"}
    Shapes:
        input_ids/labels/attention_mask/position_ids: (pack_length,) per bin (last bin <= pack_length)
        seq_lens: List[int], boundaries within the bin
    """
    raise NotImplementedError(
        "TODO(01.optional.packing): greedy concat with position_id resets; study contamination."
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_step(
    model: Any,
    micro_batches: Iterable[Any],
    optimizer: Any,
    scheduler: Any,
    accumulation_steps: int,
    max_grad_norm: float,
) -> Tuple[float, float]:
    """One optimizer step worth of gradient accumulation, clipping, and stepping.

    Contract:
      - For each micro-batch: compute_loss(model, b) / accumulation_steps -> backward().
        Accumulate the UN-scaled loss values for logging only.
      - After accumulation_steps micro-batches (or exhaustion): clip_grad_norm_
        (model.parameters(), max_grad_norm), optimizer.step(), scheduler.step(),
        optimizer.zero_grad(set_to_none=True).
      - Skip/handle non-finite losses explicitly (decide & document the policy).
    Returns:
        (avg_loss_this_step, grad_norm_before_clip)
    Shapes:
        n/a — scalars out, zero optimizer-side logic written for you.
    """
    raise NotImplementedError(
        "TODO(01.loop): accumulate scaled backward passes, clip, step opt+scheduler, reset grads."
    )


def evaluate_val_loss(model: Any, val_loader: Any) -> float:
    """Mean masked CE over the validation split under eval-mode/no_grad.

    Returns a single float (token-mean over non-ignored positions).
    """
    raise NotImplementedError("TODO(01.eval): no_grad loop over val_loader, aggregate loss.")


def generation_panel(
    model: Any,
    tokenizer: Any,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Render the fixed panel prompts and generate transcripts greedily or sampled.

    Contract:
      - Default prompts = DEFAULT_SAMPLE_PANEL_PROMPTS (the fixed 6-prompt set used for
        step-0 vs final comparison panels).
      - Chat-template render (add_generation_prompt=True), truncate to fit context.
      - do_sample=False => greedy; True => temperature/top_p sampling; always
        `pad_token_id` set so open-ended batching warns nothing.
      - Restore prior train/eval mode afterwards.
    Returns the decoded transcripts, special tokens included (you want to SEE <|eot|>).
    """
    raise NotImplementedError("TODO(01.panel): templated generate over the fixed prompt set.")


def run_training_loop(cfg: Any, model: Any, tokenizer: Any) -> Any:
    """Full SFT run: epochs x dataloader calling train_step, logging JSONL metrics.

    Contract:
      - Builds dataloader(s) via make_dataloader, optimizer AdamW(lr=cfg.lr,
        weight_decay=cfg.weight_decay), linear warmup->decay scheduler over total steps
        (mirroring warmup_ratio semantics), seed_everything(cfg.seed) FIRST.
      - Every optimizer step appends one JSONL line (metrics_path under cfg.output_dir):
          {"step", "epoch", "loss", "grad_norm", "lr"}.
      - Every cfg.sample_every steps + at step 0 + final: generation_panel snapshots.
      - Periodic evaluate_val_loss per cfg.eval_every_steps.
      - bf16 autocast around forward/backward when cfg.bf16 (device-appropriate);
        gradient checkpointing enabled per flag. Resume support comes later (labs 06/07).
    Returns/Saves:
        metrics jsonl path; checkpoints under cfg.output_dir per cfg.save_every_steps.
    """
    raise NotImplementedError(
        "TODO(01.loop.main): epochs, scheduler, autocast, metrics.jsonl, periodic panels."
    )
