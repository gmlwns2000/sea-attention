import torch
import torch.nn.functional as F
from torch import nn, optim
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from ..hf_bert import BertConfig
from .config import PerlinAttentionConfig, get_default_config
from ..common.lora import LoraLinear, lora_forward, lora_forward_linear, lora_forward_lora
import warnings
from performer_pytorch import FastAttention
from ..common.performer import ProjectionUpdater
from ..common.kl_div_for_atten import kl_div_attention

@dataclass
class PerlinAttentionOutput:
    loss: torch.Tensor
    context_layer: torch.Tensor
    partial_attention_probs: torch.Tensor
    estimated_attention_probs: torch.Tensor
    dense_attention_probs: torch.Tensor
    key_for_score: torch.Tensor

class PerlinAttention(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        perlin_config: PerlinAttentionConfig = None,
    ):
        super().__init__()
    
        self.config = config
        self.pconfig = perlin_config if perlin_config is not None else get_default_config()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        ### Perlin
        #- configs
        self.benchmarking = False
        
        #- attention predictor
        #-- mlp predictor
        self.perlin_performer_nb_features = int(
            self.attention_head_size * math.log(self.attention_head_size) / self.pconfig.performer_nb_factor
        )
        self.perlin_performer = FastAttention(
            dim_heads = self.attention_head_size,
            nb_features = self.perlin_performer_nb_features,
            causal=False,
            # no_projection=True,
        )
        self.perlin_performer_proj_updater = ProjectionUpdater(
            self.perlin_performer, 
            1000,
        )
        self.perlin_attention_predictor_enc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.all_head_size*2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        self.perlin_attention_predictor_dec_row = nn.Sequential(
            nn.Linear(config.hidden_size, self.num_attention_heads * self.pconfig.attention_predictor_length),
        )
        self.perlin_attention_predictor_dec_scaler = nn.Sequential(
            nn.Linear(config.hidden_size, self.num_attention_heads),
        )
        #-- compressed predictor
        self.perlin_attention_predictor_comp_length = \
            self.pconfig.attention_predictor_comp_patch_count * self.pconfig.attention_predictor_comp_patch_size
        self.perlin_attention_predictor_comp_codebook = nn.Parameter(
            torch.randn((self.pconfig.attention_predictor_comp_book_size, self.pconfig.attention_predictor_comp_patch_size))
        )
        self.perlin_attention_predictor_comp_enc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.all_head_size*2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        self.perlin_attention_predictor_comp_dec_row = nn.Sequential(
            nn.Linear(
                config.hidden_size, 
                self.num_attention_heads\
                    * self.pconfig.attention_predictor_comp_book_size\
                    * self.pconfig.attention_predictor_comp_patch_count
            ),
        )
        #-- TODO VQVAE
        
        #- output
        self.perlin_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.all_head_size*2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.perlin_out_random_lookup = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.all_head_size*3, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.perlin_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        q_for_atten: torch.Tensor,
        k_for_atten: torch.Tensor,
        v_for_atten: torch.Tensor,
        q_for_score: torch.Tensor,
        k_for_score: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_scores_truth: torch.Tensor,
        context_layer_truth: torch.Tensor,
    ):
        N, H, T, HID = q.shape
        v_mask = (attention_mask[:,:,:1,:].transpose(-1, -2) > -1)
        v = v * v_mask
        v_for_atten = v_for_atten * v_mask
        
        # self.perlin_performer_proj_updater.redraw_projections(q.device)
        if not self.benchmarking:
            with torch.autocast('cuda', torch.float32):
                q_type = q_for_atten.dtype
                performer_context_layer = self.perlin_performer(
                    q_for_atten, 
                    k_for_atten, 
                    v_for_atten
                )
                if q_type != performer_context_layer.dtype:
                    performer_context_layer = performer_context_layer.to(q_type)
        else:
            # TODO: fix numerical stability...
            performer_context_layer = self.perlin_performer(
                q_for_atten, 
                k_for_atten, 
                v_for_atten
            )
        performer_value = torch.cat([performer_context_layer, v], dim=-1)
        N, H, T, HID = performer_value.shape
        # (N, H, T, P)
        
        # estimate attention scores
        if self.pconfig.attention_predictor_method == 'mlp':
            t_attention_predictor = self.perlin_attention_predictor_enc(performer_value.permute(0,2,1,3).reshape(N, T, H*HID))
            estimated_attention_score = self.perlin_attention_predictor_dec_row(t_attention_predictor)\
                .view(N, T, H, -1).permute(0, 2, 1, 3)
        elif self.pconfig.attention_predictor_method == 'comp':
            warnings.warn('attention prediction method is compressed one.')
            t_attention_predictor = self.perlin_attention_predictor_comp_enc(performer_value.permute(0,2,1,3).reshape(N, T, H*HID))
            estimated_attention_score = self.perlin_attention_predictor_comp_dec_row(t_attention_predictor)\
                .view(N, T, H, -1).permute(0, 2, 1, 3)
            estimated_attention_score = estimated_attention_score\
                .view(N, H, T, self.pconfig.attention_predictor_comp_patch_count, self.pconfig.attention_predictor_comp_book_size)
            _, _, _, CODE_SEQ_LEN, BOOK_LEN = estimated_attention_score.shape
            estimated_attention_score = torch.softmax(estimated_attention_score, dim = -1)
            estimated_attention_score = torch.matmul(
                estimated_attention_score.view(-1, BOOK_LEN), 
                self.perlin_attention_predictor_comp_codebook
            )
            estimated_attention_score = estimated_attention_score.view(N, H, T, -1)
        else:
            raise Exception()
        
        # interpolate and convert to probability
        original_dtype = estimated_attention_score.dtype
        
        if torch.get_autocast_gpu_dtype() == torch.bfloat16: # F interpolate is not supported on bf16
            with torch.autocast('cuda', torch.float16):
                if estimated_attention_score.dtype != torch.float16:
                    estimated_attention_score = estimated_attention_score.to(torch.float16)
                interp_mode = 'bilinear' if T >= estimated_attention_score.shape[-1] else 'area'
                estimated_attention_score = F.interpolate(estimated_attention_score, (T, T), mode=interp_mode)
            if estimated_attention_score.dtype != original_dtype:
                estimated_attention_score = estimated_attention_score.to(original_dtype)
        else:
            interp_mode = 'bilinear' if T >= estimated_attention_score.shape[-1] else 'area'
            estimated_attention_score = F.interpolate(estimated_attention_score, (T, T), mode=interp_mode)
        
        estimated_attention_score = estimated_attention_score + attention_mask
        estimated_attention_probs = torch.softmax(estimated_attention_score, -1)
        
        # in layerwise, train perlin attention predictor
        if not self.benchmarking:
            with torch.autocast('cuda', torch.float32):
                loss = kl_div_attention(
                    F.log_softmax(estimated_attention_score, dim=-1),
                    F.softmax(attention_scores_truth, dim=-1),
                    attention_mask,
                ) * 0.1 + F.mse_loss(estimated_attention_score, attention_scores_truth)
        else:
            loss = 0
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_dense = torch.matmul(q_for_score, k_for_score.transpose(-1, -2))
        attention_scores_dense = attention_scores_dense / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores_dense = attention_scores_dense + attention_mask
        if not self.benchmarking:
            loss += F.mse_loss(attention_scores_dense, attention_scores_truth)
            attention_probs_dense = torch.softmax(attention_scores_dense, dim=-1)
        else:
            attention_probs_dense = None
        
        k = min(max(int(self.pconfig.k), 1), T)
        k_flatten = self.pconfig.k_flatten
        if not k_flatten:
            value, indices = torch.topk(
                estimated_attention_probs, # estimation gradient is cut here
                k=k, dim=-1,
            )
            # warnings.warn(f'topk({k}/{estimated_attention_probs.shape[-1]})')
            # (N, H, T, T)
            partial_attention_scores = attention_scores_dense
            partial_attention_scores_gathered = partial_attention_scores.gather(-1, indices)
            partial_attention_scores = torch.empty_like(attention_scores_truth).fill_(-10000)
            partial_attention_scores.scatter_(
                dim=-1, index=indices, src=partial_attention_scores_gathered
            )
        else:
            # (N, H, T, T)
            partial_attention_scores = attention_scores_dense
            N, H, T, T = partial_attention_scores.shape
            partial_attention_scores = partial_attention_scores.view(N, H, T*T)
            value, indices = torch.topk(
                estimated_attention_probs.view(N, H, T*T), # estimation gradient is cut here
                k=k*T, dim=-1
            )
            # warnings.warn(f'topk({k*T}/{T*T})')
            partial_attention_scores_gathered = partial_attention_scores.gather(-1, indices)
            partial_attention_scores = torch.empty_like(partial_attention_scores).fill_(-10000)
            partial_attention_scores.scatter_(
                dim=-1, index=indices, src=partial_attention_scores_gathered
            )
            partial_attention_scores = partial_attention_scores.view(N, H, T, T)
        
        if attention_mask is not None:
            partial_attention_scores = partial_attention_scores + attention_mask
        estimated_scale = self.perlin_attention_predictor_dec_scaler(t_attention_predictor).view(N, T, H, -1).permute(0, 2, 1, 3)
        partial_attention_probs = torch.softmax(partial_attention_scores, -1)
        partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scale)
        
        partial_context_layer = torch.matmul(partial_attention_probs, v)
        
        if self.pconfig.random_lookup:
            # lookup randomly that not looked up by partial context
            num_lookups = self.pconfig.random_lookup_count
            lookups = None
            estimated_attention_probs_masked = estimated_attention_probs * (attention_mask > -1) * (partial_attention_scores > -9999)
            for n in range(num_lookups):
                token_length = (attention_mask.view(N, T) > -1).float().sum(dim=-1).view(N, 1, 1, 1)
                # N, H, T, HID
                random_context_index = torch.rand_like(partial_context_layer)
                random_context_index = (random_context_index * (1 - 1/T) * token_length).floor().long()
                
                random_context_layer = v.gather(dim=-2, index=random_context_index)
                random_context_weight = estimated_attention_probs_masked.gather(dim=-1, index=random_context_index)
                random_context_layer = random_context_weight * random_context_layer
                if lookups is None:
                    lookups = random_context_layer
                else:
                    lookups = lookups + random_context_layer
            
            random_context_layer = random_context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = random_context_layer.size()[:-2] + (self.all_head_size,)
            random_context_layer = random_context_layer.view(new_context_layer_shape)

        partial_context_layer = partial_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = partial_context_layer.size()[:-2] + (self.all_head_size,)
        partial_context_layer = partial_context_layer.view(new_context_layer_shape)
        performer_context_layer = performer_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = performer_context_layer.size()[:-2] + (self.all_head_size,)
        performer_context_layer = performer_context_layer.view(new_context_layer_shape)
        
        if not self.pconfig.random_lookup:
            partial_context_layer = self.perlin_out(torch.cat([
                partial_context_layer, 
                performer_context_layer
            ], dim=-1)) + partial_context_layer
        else:
            partial_context_layer = self.perlin_out_random_lookup(torch.cat([
                partial_context_layer, 
                performer_context_layer,
                random_context_layer,
            ], dim=-1)) + partial_context_layer
        partial_context_layer = self.perlin_norm(partial_context_layer)
        
        # in layerwise train only norm
        if not self.benchmarking:
            loss += F.mse_loss(context_layer_truth, partial_context_layer)

        return PerlinAttentionOutput(
            loss=loss,
            context_layer=partial_context_layer,
            partial_attention_probs=partial_attention_probs,
            estimated_attention_probs=estimated_attention_probs,
            dense_attention_probs=attention_probs_dense,
            key_for_score=k_for_score,
        )