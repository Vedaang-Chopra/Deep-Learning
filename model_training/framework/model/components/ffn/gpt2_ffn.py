import torch 
import torch.nn as nn



class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class MLP(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.embd_dim = config.embd_dim
        
        
        self.c_fc = nn.Linear(self.embd_dim, 4* self.embd_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4* self.embd_dim, self.embd_dim)
        # self.c_proj.NANOGPT_SCALE_INIT=1

    def forward(self, x):
        x = self.c_fc(x)
        x= self.gelu(x)
        x = self.c_proj(x)
        return x
