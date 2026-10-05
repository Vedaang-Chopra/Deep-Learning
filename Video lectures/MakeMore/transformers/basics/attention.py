### Implementing the Self Attention Block

from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F



### If I initally start attention block as 0, then we do an averaaging operation and each word has equal weight. I need to in data dependent way.  Self Attention does that

'''
Basically for one sentence of context_length e.g. 8 ; I want to understand what is the relation between these 8*8 words, what is pattern, affinities, relation. 

We want tokens to learn from past. 
It solves this where each token emits 2 vectors. Query and Key
Query -> What am i looking for
Key -> What do I contain 

How to get affinities, Dot product of Key and Query, becomes the relation, if your tokens align, they will interact very high

# learned_relation = (q@k) @ x

Here we add this value, to x; rather than directly information 
V tells the private information of X.
    
So attention is (Q*K) *V ; we don't use input

# So 5th token Q -> Here is what i am interested in ; K -> here is what I have; V -> here is what i will communicate 
    
# V -> Communicates the X information 



Assume attention is basically a communication mechanism, where nodes directed graph, every node has information and that gets aggregated with nodes that point to it. 
Our grpah is serial 
1-> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

And connections between as we go ahead.
Positional encoding adds information as attention is just tokens communicate, there is no positional information there. 


Sentiment analysis wants where all tokens talk to each other
Encoder v/s decoder only difference is tril & masked_fill operation


Why sqrt d ; Why do that K and Q are unit variance gaussian(weights are normal in nature), that means mean 0 and std; 

If we multiply with sqrt d the variance becomes order of head size, 
Because of softmax, if we have very large numbers, then we will not get good values, like one hot vectors, and then we will lose information 

scaling is used to control variance -> Stability at initialization
'''

'''


Notes:
- Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
- There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other
- In an "encoder" attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- "Scaled" attention additional divides `wei` by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. Illustration below

'''


class Head(nn.Module):
    """ one head of self-attention 
    
    Input -> Give it head size, it will generate, the Q, K, V metrics
    tril is getting registered here. 
    """

    def __init__(self, n_embed,block_size, head_size, dropout, device):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.query = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.value = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        # We are scaling it and making sure stable attention happens
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out




'''
Multihead Attention - Multiple Single attentions but in parallel

'''

class MultiHeadAttention(nn.Module):
    '''
    Multiple heads of self attention in parallel 
    Multiple attentions in parallel
    
    '''
    def __init__(self, num_heads, 
                n_embed,block_size, 
                head_size, 
                dropout,
                device, 
                *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        ## Creating multiple heads here, run this list in parallel
        self.heads = nn.ModuleList([Head(n_embed,
                                        block_size,
                                        head_size, 
                                        dropout,
                                        device)for _ in range(num_heads)])
        ## Concatenating them over channels 
        self.proj = nn.Linear(head_size*num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Concat over channel dimensions
        out = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.dropout(self.proj(out))
        return out
    


        
    
    
    
    
def attention_head(n_embed, context_len, head_size, x, device):
    
    key = nn.Linear(n_embed, head_size, bias = False, device = device)
    query = nn.Linear(n_embed, head_size, bias = False, device= device)
    value = nn.Linear(n_embed, head_size, bias = False, device= device)
    # print(key, query)
    
    k = key(x)     ## (B, T, emd_dimension)
    q = query(x)    ## (B, T, emd_dimension)
    
    # print(k.shape)
    # print(q)
    ## here to do proper matrix multiplication q @ k (need to proper dimensions)
    # (B, T, C) @ (B, C, T) -> (B, T, T)
    wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
    
    tril = torch.tril(torch.ones(context_len, context_len)).to(device)
    
    ## Because of the tril and The masked fill ensures the past is only looked at. Otherwise we see the relation of all words with each other. we only allow to see the past tokens because of these 2 operations
    
    wei = wei.masked_fill(tril==0, float('-inf'))
    wei= F.softmax(wei, dim=1)
    
    v = value(x)
    print(wei.shape, v.shape)
    learned_relation = wei @ v
    
    ## Now we also create a value
    
    
    
    return learned_relation
