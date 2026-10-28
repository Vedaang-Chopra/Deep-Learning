import torch



def extract_loss_logits(model_output):
    """Normalize local GPT and Hugging Face outputs to ``(loss, logits)``."""
    loss = getattr(model_output, "loss", None)
    logits = getattr(model_output, "logits", None)

    if isinstance(model_output, dict):
        loss = model_output.get("loss", loss)
        logits = model_output.get("logits", logits)
    elif torch.is_tensor(model_output):
        logits = model_output
    elif isinstance(model_output, (tuple, list)):
        # The local model returns (loss, logits). Hugging Face tuple mode
        # returns logits first without labels, and loss then logits with labels.
        for value in model_output:
            if torch.is_tensor(value):
                if value.ndim == 3 and logits is None:
                    logits = value
                elif value.ndim == 0 and loss is None:
                    loss = value

    if logits is None or not torch.is_tensor(logits) or logits.ndim != 3:
        raise TypeError(
            "Expected model output to contain rank-3 logits as `.logits`, "
            "a `logits` mapping entry, or a tensor in its output tuple."
        )

    return loss, logits


def generate_prob(token_input, context_len, model):
    
    ## Cropping for context Length
    idx_cropped = token_input[:, -context_len:]
    with torch.no_grad():
        _, logits = extract_loss_logits(model(idx_cropped))
    ## Focus only on the last time step
    ## (Batch, n_tokens, vocab_size) -> (batch, vocab_size)
    logits = logits[:, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    return probs

def gen_argmax(model, input_batch, max_new_tokens, context_len):
    
    while input_batch.size(1) < max_new_tokens:
        probs = generate_prob(input_batch, context_len, model)    
        next_token = torch.argmax(probs, dim=-1, keepdim = True)    
        input_batch = torch.cat((input_batch, next_token), dim=1)
        
    return input_batch


def gen_sample(model, input_batch, max_new_tokens, context_len):
    
    while input_batch.size(1) < max_new_tokens:
            probs = generate_prob(input_batch, context_len, model)    
            topk_probs , topk_indices = torch.topk(probs, 50, dim=-1)
            next_token = torch.multinomial(topk_probs, num_samples = 1)
            xcol = torch.gather(topk_indices, -1, next_token)
            input_batch = torch.cat((input_batch, xcol), dim=1)
    return input_batch
