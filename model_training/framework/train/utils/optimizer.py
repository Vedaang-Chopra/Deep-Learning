
import torch 
import inspect



## Setting up Weight Decay
def configure_optimizer(model, weight_decay, learning_rate, device, master_process, eps=1e-8, betas=(0.9, 0.95)):
    
    ### Step-1: - Start with all of the parameters (that require grad)
    #  Separate parameters that have to be updated v/s not updated
    # start with all of the candidate parameters (that require grad)
    ## Example: - Biases not need weight decay, or 1-D tensors layer norms
    #  Focus on more Matrix Multiplication weights and embeddings
    # Like Regularization , where we add |W|^2 ; but W are of certain layers
    
    
    # Identify all weights that are to be updated
    param_dict = {pn:p for pn, p in model.named_parameters()}
    param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
    
    
    decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
    non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': non_decay_params, 'weight_decay': 0.0},
        
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_non_decay_params = sum(p.numel() for p in non_decay_params)
    if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {num_non_decay_params:,} parameters")
    
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"
    if master_process:
            print(f"using fused AdamW: {use_fused}")
   
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused)
    return optimizer
