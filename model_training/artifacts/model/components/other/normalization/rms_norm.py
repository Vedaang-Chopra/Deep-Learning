
import torch
from torch import Tensor, nn

class RMSNorm(nn.Module):
    
    def __init__(self, dim:int , eps:float = 1e-6):
        self.eps = eps              ## EPS to avoid division by 0
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x:torch.Tensor):
        ## Shape of X: - (B, Seq_Len, Dim)
        ## rsqrt = 1 / sqrt
        return x * torch.rsqrt(self.eps + x.pow(2).mean(-1, keepdim=True))
    
    def forward(self, x:Tensor)-> Tensor:
        ## (1, 1, Dim) or Dim * (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        return (self.weight * self._norm(x.float())).type_as(x)
    