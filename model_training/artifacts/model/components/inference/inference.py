import torch



def generate_prob(token_input, context_len, model):
    
    ## Cropping for context Length
    idx_cropped = token_input[:, -context_len:]
    with torch.no_grad():
        loss, logits = model(idx_cropped)
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