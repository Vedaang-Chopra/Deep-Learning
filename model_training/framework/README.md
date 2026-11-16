# framework

Reusable pieces for pre-training and post-training experiments. This is a learning project, not a production library. The notebooks in `../pre_training/` and `../post_training/` import from here.

## Layout

```
framework/
├── model/
│   ├── components/
│   │   ├── attention/    MHA, GQA, MQA, MLA, sliding window
│   │   ├── ffn/          GPT-2 MLP, SwiGLU, SigLIP FFN
│   │   ├── other/        RoPE, embeddings, normalization (Layer/RMS/Batch)
│   │   ├── block/        GPT-2, LLaMA, PaliGemma, SigLIP encoder blocks
│   │   └── inference/    generation helpers
│   ├── load_model/       GPT-2, SigLIP, PaliGemma configs and models
│   └── tokenizer/
├── train/
│   ├── core/             training loop, config, multi-GPU setup
│   ├── utils/            optimizer, LR scheduler
│   └── tracking/         experiment logger
└── data/
    ├── pre_train/        TinyShakespeare token-stream loader
    └── post_train/       SFT, classification (spam, Banking77), RLHF
```

## Status

- Models built from these parts: GPT-2, LLaMA-style decoder, SigLIP, PaliGemma (in progress).
- Not done yet: components take the whole model config, and blocks hardcode their attention, FFN and norm choices, so swapping a component means editing the block.
- Next: a uniform component interface, then blocks built from named choices (for example `attn="gqa", ffn="swiglu"`).
- Known issues: a syntax error in `attention/variants/mqa.py`; `block/llama_block.py` has missing imports.
- No automated tests yet.

## Planned

Objectives for DPO, PPO and GRPO that plug into one generic trainer, rollout generation with a KV cache, and parity tests against the existing models.
