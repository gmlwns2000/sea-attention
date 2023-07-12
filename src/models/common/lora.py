from torch import nn
import torch
import torch.nn.functional as F
import math

class LoraLinear(nn.Module):
    def __init__(self, inch, outch, dim_r):
        super().__init__()
        
        self.lora_a = nn.Parameter(torch.zeros((dim_r, inch)))
        self.lora_b = nn.Parameter(torch.zeros((outch, dim_r)))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor):
        x = F.linear(x, self.lora_a)
        x = F.linear(x, self.lora_b)
        return x

def lora_forward(linear: nn.Linear, lora: LoraLinear, x: torch.Tensor, enabled: bool):
    if not enabled:
        return linear(x)
    assert linear.bias.ndim == 1
    assert x.ndim == 3
    
    x_fc = F.linear(x, linear.weight)
    x_lora = lora(x)
    x = x_fc + x_lora
    if linear.bias is not None:
        x = x + linear.bias.view(1, 1, linear.bias.shape[0])
    return x

# split for save memory
def lora_forward_linear(linear: nn.Linear, x: torch.Tensor):
    return F.linear(x, linear.weight, linear.bias)

def lora_forward_lora(linear: nn.Linear, linear_x: torch.Tensor, lora: LoraLinear, x: torch.Tensor, enabled: bool):
    if not enabled:
        return linear_x
        # assert linear.bias.ndim == 1
        # assert linear_x.ndim == 3
        # if linear.bias is not None:
        #     return linear_x + linear.bias.view(1, 1, linear.bias.shape[0])
        # return linear_x
    
    if linear.bias is not None:
        linear_x = linear_x - linear.bias.view(1, 1, linear.bias.shape[0])
    lora_x = lora(x)
    x = lora_x + linear_x
    if linear.bias is not None:
        return x + linear.bias.view(1, 1, linear.bias.shape[0])
    return x