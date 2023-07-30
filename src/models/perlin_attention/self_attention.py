import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim

from ..common.lora import (
    LoraLinear, 
    lora_forward, 
    lora_forward_linear,
    lora_forward_lora
)
from ..hf_bert import BertConfig
from .attention import PerlinAttention, PerlinAttentionOutput
from .config import PerlinAttentionConfig, get_default_config

default_lazy = lambda x, d: d() if x is None else x

class PerlinSelfAttention(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        perlin_config: PerlinAttentionConfig = None,
    ):
        super().__init__()
        
        self.config = config
        self.pconfig = perlin_config if perlin_config is not None else get_default_config()
        # warnings.warn(f'PerlinSelfAttentionConfig: {self.pconfig}')

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.last_loss = None

        self.query_lora = LoraLinear(config.hidden_size, self.all_head_size, self.pconfig.lora_r)
        self.key_lora = LoraLinear(config.hidden_size, self.all_head_size, self.pconfig.lora_r)
        self.value_lora = LoraLinear(config.hidden_size, self.all_head_size, self.pconfig.lora_r)
        
        self.query_lora_for_approx_score = LoraLinear(config.hidden_size, self.all_head_size, self.pconfig.lora_r)
        self.key_lora_for_approx_score = LoraLinear(config.hidden_size, self.all_head_size, self.pconfig.lora_r)
        
        self.query_lora_for_approx_atten = LoraLinear(config.hidden_size, self.all_head_size, self.pconfig.lora_r)
        self.key_lora_for_approx_atten = LoraLinear(config.hidden_size, self.all_head_size, self.pconfig.lora_r)
        self.value_lora_for_approx_atten = LoraLinear(config.hidden_size, self.all_head_size, self.pconfig.lora_r)
        
        self.attention = PerlinAttention(
            config=config,
            perlin_config=perlin_config,
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4: return x
        
        assert x.ndim == 3
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: nn.Module,
        key: nn.Module,
        value: nn.Module,
        hidden_states: torch.Tensor = None,
        query_layer: torch.Tensor = None,
        key_layer: torch.Tensor = None,
        value_layer: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_scores_truth: Optional[torch.FloatTensor] = None,
        context_layer_truth: Optional[torch.FloatTensor] = None,
        last_state: object = None,
    ) -> Tuple[torch.Tensor]:
        if self.pconfig.layerwise and self.training:
            hidden_states = hidden_states.detach()
        
        t_key_layer = default_lazy(key_layer, lambda: lora_forward_linear(key, hidden_states))
        key_layer = self.transpose_for_scores(lora_forward_lora(
            linear=key, 
            linear_x=t_key_layer, 
            lora=self.key_lora, 
            x=hidden_states, 
            enabled=self.pconfig.lora_enabed
        ))
        if self.pconfig.lora_in_approx_enabled:
            key_layer_for_atten = self.transpose_for_scores(lora_forward_lora(
                linear=key, 
                linear_x=t_key_layer, 
                lora=self.key_lora_for_approx_atten, 
                x=hidden_states, 
                enabled=self.pconfig.lora_in_approx_enabled
            ))
            key_layer_for_score = self.transpose_for_scores(lora_forward_lora(
                linear=key, 
                linear_x=t_key_layer, 
                lora=self.key_lora_for_approx_score, 
                x=hidden_states, 
                enabled=self.pconfig.lora_in_approx_enabled
            ))
        else:
            key_layer_for_atten = key_layer_for_score = key_layer
        
        t_value_layer = default_lazy(value_layer, lambda: lora_forward_linear(value, hidden_states))
        value_layer = self.transpose_for_scores(lora_forward_lora(
            linear=value, 
            linear_x=t_value_layer, 
            lora=self.value_lora, 
            x=hidden_states, 
            enabled=self.pconfig.lora_enabed
        ))
        if self.pconfig.lora_in_approx_enabled:
            value_layer_for_atten = self.transpose_for_scores(lora_forward_lora(
                linear=value, 
                linear_x=t_value_layer, 
                lora=self.value_lora_for_approx_atten, 
                x=hidden_states, 
                enabled=self.pconfig.lora_in_approx_enabled
            ))
        else:
            value_layer_for_atten = value_layer
        
        t_mixed_query_layer = default_lazy(query_layer, lambda: lora_forward_linear(query, hidden_states))
        query_layer = self.transpose_for_scores(lora_forward_lora(
            linear=query, 
            linear_x=t_mixed_query_layer, 
            lora=self.query_lora, 
            x=hidden_states, 
            enabled=self.pconfig.lora_enabed
        ))
        if self.pconfig.lora_in_approx_enabled:
            query_layer_for_atten = self.transpose_for_scores(lora_forward_lora(
                linear=query, 
                linear_x=t_mixed_query_layer, 
                lora=self.query_lora_for_approx_atten, 
                x=hidden_states, 
                enabled=self.pconfig.lora_in_approx_enabled
            ))
            query_layer_for_score = self.transpose_for_scores(lora_forward_lora(
                linear=query, 
                linear_x=t_mixed_query_layer, 
                lora=self.query_lora_for_approx_score, 
                x=hidden_states, 
                enabled=self.pconfig.lora_in_approx_enabled
            ))
        else:
            query_layer_for_atten = query_layer_for_score = query_layer
        
        output = self.attention(
            q=query_layer,
            k=key_layer,
            v=value_layer,
            q_for_atten=query_layer_for_atten,
            k_for_atten=key_layer_for_atten,
            v_for_atten=value_layer_for_atten,
            q_for_score=query_layer_for_score,
            k_for_score=key_layer_for_score,
            attention_mask=attention_mask,
            attention_scores_truth=attention_scores_truth,
            context_layer_truth=context_layer_truth,
            last_state=last_state,
        ) #type: PerlinAttentionOutput
        self.last_attention_probs = output.partial_attention_probs
        
        if self.pconfig.layerwise and self.training:
            output.partial_attention_probs = output.partial_attention_probs.detach()
            output.context_layer = output.context_layer.detach()
        
        return output