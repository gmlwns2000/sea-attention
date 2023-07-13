import torch
import torch.nn.functional as F
from torch import nn, optim
import math
from dataclasses import dataclass, asdict
from ..hf_bert import BertConfig
from typing import Optional, Tuple, List, Dict
import json

@dataclass
class PerlinAttentionConfig:
    performer_nb_factor: int = 1
    k: int = 7
    k_flatten: bool = True
    k_flatten_dim: str = 'batch'
    random_lookup: bool = False
    random_lookup_count: int = 3
    attention_predictor_method: str = 'mlp'
    attention_predictor_length: int = 128
    attention_predictor_comp_book_size: int = 8
    attention_predictor_comp_patch_size: int = 16
    attention_predictor_comp_patch_count: int = 16
    layerwise: bool = False
    lora_r: int = 32
    lora_enabed: bool = False
    lora_in_approx_enabled: bool = False
    partial_attention_scaler: bool = True
    
    def to_json(self):
        return asdict(self)

    def __repr__(self) -> str:
        return f"PerlinAttentionConfig({json.dumps(self.to_json())})"
    
DEFAULT_CONFIG = PerlinAttentionConfig()

def register_default_config(config: PerlinAttentionConfig):
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config
    
def get_default_config() -> PerlinAttentionConfig:
    global DEFAULT_CONFIG
    return DEFAULT_CONFIG