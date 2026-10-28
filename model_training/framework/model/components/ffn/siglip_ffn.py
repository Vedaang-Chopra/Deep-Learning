import torch 
import torch.nn as nn
from torch.nn import functional as F

class SiglipMLP(nn.Module):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        
        self.hidden_size, self.intermediate_size = config.hidden_size, config.intermediate_size
        
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
        
    def forward(self, hidden_states:torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        
        hidden_states = F.gelu(hidden_states, approximate= 'tanh')
        
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states
        