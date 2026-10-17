import torch 
import torch.nn as nn
from torch.functional import Functional as F


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

        
        self.w2.NANOGPT_SCALE_INIT = 1
        
        
    def forward(self, x: Tensor) -> Tensor:
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x= swish * x_V
        x = self.w2(x)
        return x