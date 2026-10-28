
import torch


def text_to_token_ids(text, tokenizer, allowed_special):
    encoded = tokenizer.encode(text, allowed_special=allowed_special)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(tokens, tokenizer):
    if torch.is_tensor(tokens):
        tokens = tokens.detach().cpu()
        if tokens.ndim == 2:
            if tokens.size(0) != 1:
                raise ValueError(
                    "token_ids_to_text expects one sequence; pass one batch item at a time."
                )
            tokens = tokens.squeeze(0)
        if tokens.ndim != 1:
            raise ValueError("Expected token IDs with shape [sequence] or [1, sequence].")
        tokens = tokens.tolist()

    return tokenizer.decode(tokens)
