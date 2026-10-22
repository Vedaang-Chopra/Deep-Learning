
import torch 
import torch.nn as nn
from torch.nn import functional as F

import code
import time




def train_model(model, train_components, train_config):
    
    
    if train_config.accumulation_steps < 1:
        raise ValueError("train_config.accumulation_steps must be at least 1")

    print(f"Current Device: {train_config.device}")
    model.train()
    print(f"Model set to Training mode:{model.training}")
    
    if train_config.use_compile:
        model = model.to(train_config.device)    
        model = torch.compile(model)
    else:
        model = model.to(train_config.device)    
    
    for step in range(0,(train_config.num_steps)):
        
        start_time=time.time()
        ### Step-1: - Zero the Gradients in Training 
        train_components.optimizer.zero_grad()
        
        ### Adding Gradient Accumulation -> Due to GPU limits, we can't fit all data in one GPU, so we do multiple forward and backward and then do one update after some data is processed. 
        ## Some data is batch size, so in One Batch -> Do multiple forward and backward pass and then do one update. We want Batch Size to be High, but due to GPU limit we can't do that
        
        
        ### Step-2:- Load the data on Device
        loss_accum = 0.0 ## For keeping track of values 
        for micro_step in range(0,train_config.accumulation_steps):
            x, y = train_components.dataloader.next_batch()
            x= x.to(train_config.device)
            y = y.to(train_config.device)
            
            ## Step-3a: - Do the Forward Pass
            with torch.autocast(device_type= train_config.device, dtype=torch.bfloat16):
                model_output = model(x)
                if hasattr(model_output, "logits"):
                    logits = model_output.logits  # Hugging Face output
                else:
                    _, logits = model_output      # Local GPT returns (loss, logits)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                )
            
            ## Here since we do multiple forward backward, we need to do proper average over all steps. 
            loss = loss / train_config.accumulation_steps
            loss_accum += loss.detach()
            
            
            ## Step-3b: - Send the Loss backward
            loss.backward()
        
        
        ## Gradient Clipping; to handle the shock for bad batches. Handles instability, so that model weight updates are not very high
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Setup LR with Scheduler
        lr = train_components.scheduler(step=step,  
                                                total_steps=train_config.num_steps,
                                                warmup_steps= train_config.warmup_steps,
                                                peak_lr=train_config.max_lr,
                                                min_lr= train_config.min_lr)
        
        for param_group in train_components.optimizer.param_groups:
            param_group['lr'] = lr
        
        ### Step-4: - Update the weights -> With weight Decay
        train_components.optimizer.step() 
        
        
        torch.cuda.synchronize() ### Wait for it to finish a batch
        
        
        end_time = time.time()
        dt = (end_time-start_time)*1000
        
        tokens_per_second = (train_components.dataloader.batch_size * train_config.seq_length * train_config.accumulation_steps) / (end_time-start_time)
        
        print(f"""step { step } | loss: {loss_accum.item():.6f} | norm :{norm:.2f} | dt: {dt:.2f} ms | tok/sec:{tokens_per_second:.2f}""")
        
        # Log the details
        # train_config.log()
    return model


# import torch 
# import torch.nn as nn
# from torch.nn import functional as F

# import code
# import time




# def train_model(model, train_components, train_config):
    
    print(f"Current Device: {train_config.device}")
    model.train()
    print(f"Model set to Training mode:{model.training}")
    
    if train_config.use_compile:
        model = model.to(train_config.device)    
        model = torch.compile(model)
    else:
        model = model.to(train_config.device)    
    
    for step in range(0,(train_config.num_steps)):
        
        start_time=time.time()
        ### Step-1: - Zero the Gradients in Training 
        train_components.optimizer.zero_grad()
        
        ### Adding Gradient Accumulation -> Due to GPU limits, we can't fit all data in one GPU, so we do multiple forward and backward and then do weight update after some data is processed. 
        ## Some data is batch size, so in One Batch -> Do multiple forward and backward pass and then do one update. We want Batch Size to be High, but due to GPU limit we can't do that
        
        
        ### Step-2:- Load the data on Device
        loss_accum = 0.0 ## For keeping track of values 
        for micro_step in range(0,train_config.accumulation_steps):
            x, y = train_components.dataloader.next_batch()
            x= x.to(train_config.device)
            y = y.to(train_config.device)
            
            ## Step-3a: - Do the Forward Pass
            with torch.autocast(device_type= train_config.device, dtype=torch.bfloat16):
                loss , logits= model(x, y)
                # code.interact(local=locals())
            
            ## Here since we do multiple forward backward, we need to do proper average over all steps. 
            loss = loss / train_config.accumulation_steps
            loss_accum += loss.detach()
            
            ## Step-3b: - Send the Loss backward
            loss.backward()
        
        
        ## Gradient Clipping; to handle the shock for bad batches. Handles instability, so that model weight updates are not very high
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Setup LR with Scheduler
        lr = train_components.scheduler(step=step,  
                                                total_steps=train_config.num_steps,
                                                warmup_steps= train_config.warmup_steps,
                                                peak_lr=train_config.max_lr,
                                                min_lr= train_config.min_lr)
        
        for param_group in train_components.optimizer.param_groups:
            param_group['lr'] = lr
        
        ### Step-4: - Update the weights -> With weight Decay
        train_components.optimizer.step() 
        
        
        torch.cuda.synchronize() ### Wait for it to finish a batch
        
        
        end_time = time.time()
        dt = (end_time-start_time)*1000
        
        tokens_per_second = (train_components.dataloader.batch_size * train_components.dataloader.seq_len * train_config.accumulation_steps) / (end_time-start_time)
        
        print(f"""step { step } | loss: {loss_accum.item():.6f} | norm :{norm:.2f} | dt: {dt:.2f} ms | tok/sec:{tokens_per_second:.2f}""")
        
        # Log the details
        # train_config.log()
    return model