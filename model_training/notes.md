Current Models 

Agent + Reasoning + Effort (How are this working today)

Models -> Can switch thinking on and off ? 



Objective	What the model learns	Architectural implications
Causal language modeling (CLM)	Predict the next token from previous tokens	Decoder-only, causal mask
Fill-in-the-middle (FIM)	Reconstruct a missing middle span from prefix and suffix	Compatible with a causal decoder after reordering tokens
Multi-token prediction (MTP)	Predict multiple future tokens at each position	Additional prediction heads or modules
Masked language modeling (MLM)	Recover randomly masked tokens using context on both sides	Usually bidirectional attention
Span corruption / denoising	Recover removed or corrupted text spans	Often encoder-decoder, as in T5
Contrastive learning	Align related representations and separate unrelated ones	Representation heads, paired samples; especially useful for multimodal training




Architecture progression: GPT-2 → LLaMA 2 → Mistral → Gemma 2 → Qwen3 → DeepSeek.


Order	Architecture	New components to focus on
1	GPT-2 (2019)	MHA, learned positional embeddings, LayerNorm, GELU, residual connections, weight tying
2	LLaMA 2 (2023)	RoPE, RMSNorm, SwiGLU, pre-norm, GQA in the 70B variant; introduce KV caching
3	Mistral 7B (2023)	Sliding-window attention, GQA, rolling KV cache
4	Gemma 2 (2024)	Alternating local/global attention, GeGLU, attention/logit soft-capping, additional normalization
5	Qwen3 Dense (2025)	QK-Norm, attention projection choices, modern dense Transformer configuration
6	DeepSeek-V2 (2024)	Multi-head Latent Attention (MLA), compressed KV representations, sparse MoE, expert routing
7	DeepSeek-V3 (2024)	Auxiliary-loss-free MoE balancing, multi-token prediction, more advanced training design






### Dataset 