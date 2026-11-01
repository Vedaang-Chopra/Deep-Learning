import torch
from types import SimpleNamespace
from transformers import GPT2Config as HFConfig, GPT2LMHeadModel

from model_training.framework.data.pre_train.causal_modelling.dataloader import DataLoaderLite
from model_training.framework.model.components.inference.inference import gen_argmax, gen_sample
from model_training.framework.model.load_model.gpt2.configs import GPT2Config
from model_training.framework.model.load_model.gpt2.model import GPT
from model_training.framework.train.core.config import TrainConfig
from model_training.framework.train.core.train import train_model
from model_training.framework.train.utils.lr_scheduler import cosine_lr


def tiny_model():
    return GPT(GPT2Config(vocab_size=31, context_len=8, n_layer=2, n_head=2, emb_dim=16))


def test_batch_and_training_step():
    loader = DataLoaderLite(list(range(17)), batch_size=2, seq_len=4, tokenizer=None)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (2, 4)
    assert torch.equal(y.flatten(), x.flatten() + 1)
    model = tiny_model()
    loss, logits = model(x, y)
    assert logits.shape == (2, 4, 31)
    loss.backward()
    assert model.transformer.wte.weight.grad is not None


def test_generation_counts_new_tokens_with_small_vocab():
    model = tiny_model().eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    assert gen_argmax(model, prompt, 2, 8).shape == (1, 6)
    assert gen_sample(model, prompt, 2, 8).shape == (1, 6)


def test_huggingface_weight_mapping_and_logits():
    hf = GPT2LMHeadModel(HFConfig(vocab_size=31, n_positions=8, n_ctx=8, n_embd=16, n_layer=2, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
    model = tiny_model().use_pretrained(hf).eval()
    x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.no_grad():
        actual = model(x)[1]
        expected = hf(x).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert model.lm_head.weight is model.transformer.wte.weight


def test_pretrained_factory_uses_supported_small_alias(monkeypatch):
    hf = GPT2LMHeadModel(HFConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2, n_head=2))
    seen = []
    def load_checkpoint(name):
        seen.append(name)
        return hf
    monkeypatch.setattr(GPT2LMHeadModel, "from_pretrained", load_checkpoint)
    model = GPT.from_pretrained(GPT2Config(vocab_size=31, context_len=8, n_layer=2, n_head=2, emb_dim=16), "gpt2-small")
    assert seen == ["gpt2"]
    torch.testing.assert_close(model.transformer.wte.weight, hf.transformer.wte.weight)


def test_trainer_runs_on_cpu():
    model = tiny_model()
    loader = DataLoaderLite(list(range(17)), batch_size=2, seq_len=4, tokenizer=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    components = SimpleNamespace(dataloader=loader, optimizer=optimizer, scheduler=cosine_lr)
    config = TrainConfig(num_steps=1, learning_rate=0.001, device="cpu", total_batch_size=8,
                         micro_batch_size=2, seq_length=4, accumulation_steps=1,
                         max_lr=0.001, min_lr=0.0001, warmup_steps=0, use_compile=False)
    before = model.transformer.wte.weight.detach().clone()
    train_model(model, components, config)
    assert not torch.equal(before, model.transformer.wte.weight)
