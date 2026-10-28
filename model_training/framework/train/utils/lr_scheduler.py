import math



# Cosine LR from Karpathy Video
# def get_lr(it, warmup_steps max_lr),:
#     # 1) linear warmup for warmup_iters steps
#     if it < warmup_steps:
#         return max_lr * (it+1) / warmup_steps
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > max_steps:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
#     return min_lr + coeff * (max_lr - min_lr)



def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:

    if step >= total_steps:
        return min_lr

    if warmup_steps == 0:
        progress = step / total_steps
        c = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + c * (peak_lr - min_lr)

    if step < warmup_steps:
        return peak_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    c = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + c * (peak_lr - min_lr)




