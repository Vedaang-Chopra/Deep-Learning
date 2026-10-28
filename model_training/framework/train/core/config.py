
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TrainConfig:
    num_steps:int
    learning_rate:int
    device: Any

    total_batch_size:int
    micro_batch_size: int
    seq_length: int
    accumulation_steps: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    use_compile:bool = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    seed: int = 2026