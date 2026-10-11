import torch 
import torch.nn as nn




class MLP(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_fc = nn.Linear(config.emb_dim, 4* config.emb_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4* config.emb_dim, config.emb_dim)
        # self.c_proj.NANOGPT_SCALE_INIT=1

    def forward(self, x):
        x = self.c_fc(x)
        x= self.gelu(x)
        x = self.c_proj(x)
        return x
