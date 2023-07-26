import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim

def interpolate(x: torch.Tensor, size, interp_mode: str = None):
    if x.shape[-2:] == size: return x
    
    interp_mode = ('bilinear' if size[-1] >= x.shape[-1] else 'area') if interp_mode is None else interp_mode
    
    if torch.get_autocast_gpu_dtype() == torch.bfloat16: # F interpolate is not supported on bf16
        original_dtype = x.dtype
        with torch.autocast('cuda', torch.float32):
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            x = F.interpolate(x, size, mode=interp_mode)
        if x.dtype != original_dtype:
            x = x.to(original_dtype)
    else:
        x = F.interpolate(x, size, mode=interp_mode)
    
    return x

class Residual(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.net = nn.Sequential(*args)
    
    def forward(self, x):
        y = self.net(x)
        return x + y

class KeepRes(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.net = nn.Sequential(*args)
    
    def forward(self, x):
        x_shape = x.shape
        x = self.net(x)
        x = interpolate(x, x_shape[-2:])
        return x

class ResBlock(nn.Module):
    def __init__(self, ch, padding=1, lnorm_size=None, padding_mode='zeros', causal=False):
        super().__init__()
        
        self.net = KeepRes(
            CausalConv2d(ch, ch, 3, padding=padding, padding_mode=padding_mode, causal=causal),
            # nn.BatchNorm2d(48),
            # nn.LayerNorm(lnorm_size),
            nn.ReLU(),
            CausalConv2d(ch, ch, 3, padding=padding, padding_mode=padding_mode, causal=causal),
            # nn.BatchNorm2d(48),
            # nn.LayerNorm(lnorm_size),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_out = self.net(x)
        x = self.relu(x_out + x)
        return x

class UpsampleFP32(nn.Module):
    def __init__(self, scale, dtype=torch.float32):
        super().__init__()
        self.scale = scale
        self.dtype = dtype
    
    def forward(self, x):
        x_type = x.dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if x_type != x.dtype:
            x = x.to(x_type)
        return x
    
class CausalConv2d(nn.Module):
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = 'zeros',
        causal: bool = False,
    ):
        super().__init__()
        
        self.causal = causal
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding)
        self.padding_mode = padding_mode
        
        # to follow pytorch initializer
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        w = conv2d.weight.data
        b = conv2d.bias.data
        
        self.bias = nn.Parameter(b)
        if not causal:
            self.weight = nn.Parameter(w)
        else:
            weight = torch.zeros((out_channels, in_channels, kernel_size * 2 - 1, kernel_size))
            weight[:,:,:kernel_size,:] = w
            self.weight = nn.Parameter(weight)
            
            weight_mask = torch.zeros((out_channels, in_channels, kernel_size * 2 - 1, kernel_size))
            weight_mask[:,:,:kernel_size,:] = 1.0
            self.register_buffer('weight_mask', None)
            self.weight_mask = weight_mask
            
            assert padding == (kernel_size // 2), "always same padding allowed"
            self.padding = (kernel_size-1, padding)
    
    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                input=F.pad(
                    input, 
                    (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), 
                    mode=self.padding_mode
                ),
                weight=weight, 
                bias=bias, 
                stride=self.stride,
                padding=0, 
                dilation=1, 
                groups=1
            )
        return F.conv2d(
            input=input, 
            weight=weight, 
            bias=bias, 
            stride=self.stride,
            padding=self.padding, 
            dilation=1,
            groups=1,
        )
    
    def forward(self, x: torch.Tensor):
        return self._conv_forward(
            input=x,
            weight=self.weight * self.weight_mask if self.causal else self.weight,
            bias=self.bias,
        )
