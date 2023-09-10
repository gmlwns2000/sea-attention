import json
import copy
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .masked_mm import sparse_attn
import h5py

import torch
# import torch.sparse._triton_ops
import torch.nn.functional as F
from performer_pytorch import FastAttention
from torch import nn, optim
# from xformers.components.attention.core import (
#     SparseCS,
#     scaled_dot_product_attention
# )

from ...utils import batch_to, get_bench, Metric
from ..common.kl_div_for_atten import kl_div_attention
from ..common.lora import (
    LoraLinear, 
    lora_forward, 
    lora_forward_linear,
    lora_forward_lora
)
from ..common.performer import ProjectionUpdater
from ..hf_bert import BertConfig
from .config import PerlinAttentionConfig, get_default_config
from ...utils import raise_if_nan, strify
from .modules import (
    ResBlock,
    Residual,
    KeepRes,
    CausalConv2d,
    UpsampleFP32,
    interpolate
)
from math import ceil, floor
# NOTE comment below to debug NaN
raise_if_nan = lambda x: x
# torch.autograd.set_detect_anomaly(True)

timer = lambda name: get_bench().region(name)
mem = lambda name: get_bench().mem_region(name)
metric = Metric()

def grid_sample_bf16(input, grid, mode='nearest', align_corners=False, padding_mode='zeros', output_dtype=None):
    input_dtype = input.dtype
    op_dtype = torch.float32 if torch.get_autocast_gpu_dtype() == torch.bfloat16 else input_dtype
    if op_dtype != input_dtype:
        input = input.to(op_dtype)
        grid = grid.to(op_dtype)
    y = F.grid_sample(
        input=input,
        grid=grid,
        mode=mode,
        align_corners=align_corners,
        padding_mode='zeros',
    )
    if output_dtype is not None:
        input_dtype = output_dtype
    if y.dtype != input_dtype:
        y = y.to(input_dtype)
    return y
# def grid_sample_bf16(input, grid, mode='nearest', align_corners=False, padding_mode='zeros'):
#     input_dtype = input.dtype
#     op_dtype = torch.float32 if torch.get_autocast_gpu_dtype() == torch.bfloat16 else input_dtype
#     if op_dtype != input_dtype:
#         input = input.to(op_dtype)
#         grid = grid.to(op_dtype)
#     y = F.grid_sample(
#         input=input,
#         grid=grid,
#         mode=mode,
#         align_corners=align_corners,
#         padding_mode='zeros',
#     )
#     if y.dtype != input_dtype:
#         y = y.to(input_dtype)
#     return y

def softmax_bf16(input, dim=-1):
    input_dtype = input.dtype
    op_dtype = torch.float32 if torch.get_autocast_gpu_dtype() in [torch.bfloat16, torch.float16] else input_dtype
    if op_dtype != input_dtype:
        input = input.to(op_dtype)
    y = torch.softmax(input, dim=-1)
    if y.dtype != input_dtype:
        y = y.to(input_dtype)
    return y

def resize_width(img: torch.Tensor, scale: float):
    N, H, W = img.shape
    # img = img.coalesce() # TODO why not using this?
    idx = img.indices() #.float()
    nnz = idx.shape[-1]
    if scale < 1.0:
        xs_scaled = idx[2] * scale
        xs_rounded = torch.clamp(torch.round(xs_scaled), 0, round(W*scale)-1)
        xs_okay = torch.abs(xs_rounded - xs_scaled) > (scale * 0.5)
        idx[2] = xs_rounded# * xs_okay
        # idx = idx * xs_okay.unsqueeze(0) + (~xs_okay.unsqueeze(0)) * -1
        idx.masked_fill_(xs_okay, value=-1)
        idx = torch.unique(idx, dim=-1) #TODO FIX this to masked select
        idx = idx[:, 1:]
        
        # print(nnz, idx.shape[-1])
        return torch.sparse_coo_tensor(
            indices=idx.contiguous(),
            values=torch.ones((idx.shape[-1],), device=img.device, dtype=img.dtype),
            size=(N, H, round(W*scale)),
        )
    elif scale > 1.0:
        scale_ceil = math.ceil(scale)
        # idx = F.interpolate(idx.view(1, 1, 3, nnz), size=(3, nnz*scale_ceil), mode='nearest').view(3, nnz*scale_ceil) # type: torch.Tensor
        idx = idx.view(3, nnz, 1).expand(3, nnz, scale_ceil).reshape(3, nnz*scale_ceil)
        # idx[2] = idx[2] * scale_ceil + torch.arange(nnz*scale_ceil, device=img.device) % scale_ceil
        
        shrink_scale = scale / scale_ceil
        xs_scaled = (idx[2] * scale_ceil + torch.arange(scale_ceil, device=img.device).unsqueeze(0).expand(nnz, scale_ceil).reshape(-1)) * shrink_scale
        xs_rounded = torch.round(xs_scaled)
        xs_okay = torch.abs(xs_rounded - xs_scaled) < (shrink_scale * 0.5)
        del xs_scaled
        idx[2] = torch.clamp(xs_rounded, 0, round(W*scale)-1)
        # idx = idx * xs_okay.unsqueeze(0) + (~xs_okay.unsqueeze(0)) * -1
        # idx.masked_fill_(xs_okay, value=-1)
        # del xs_okay
        # idx = torch.unique(idx, dim=-1)
        # idx = idx[:, 1:]
        # idx = torch.unique(idx.long(), dim=-1)
        idx = idx.masked_select(xs_okay).view(3, -1)
        
        # print(nnz, idx.shape[-1])
        return torch.sparse_coo_tensor(
            indices=idx,
            values=torch.ones((1,), device=img.device, dtype=img.dtype).expand(idx.shape[-1]),
            size=(N, H, round(W*scale)),
        )
    else:
        return img
from .attention_state import PerlinAttentionState, StatefulCausalCNN, StatefulCausalPerformer

def safe_to(t, d):
    if isinstance(t, torch.Tensor):
        return t.to(d)
    return t

from typing import NamedTuple

# @dataclass
class PerlinAttentionOutput(NamedTuple):
    loss: torch.Tensor
    context_layer: torch.Tensor
    partial_attention_probs: torch.Tensor
    partial_attention_mask: torch.Tensor
    estimated_attention_probs_m: torch.Tensor
    estimated_attention_probs: torch.Tensor
    dense_attention_probs: torch.Tensor
    key_for_score: torch.Tensor
    state: PerlinAttentionState
    
    def to(self, device):
        return PerlinAttentionOutput(
            loss=safe_to(self.loss, device),
            context_layer=safe_to(self.context_layer, device),
            partial_attention_probs=safe_to(self.partial_attention_probs, device),
            partial_attention_mask=safe_to(self.partial_attention_mask, device),
            estimated_attention_probs_m=safe_to(self.estimated_attention_probs_m, device),
            estimated_attention_probs=safe_to(self.estimated_attention_probs, device),
            dense_attention_probs=safe_to(self.dense_attention_probs, device),
            key_for_score=safe_to(self.key_for_score, device),
            state=safe_to(self.state, device),
        )

class ModuleBenchmark(nn.Module):
    def __init__(self, name, module):
        super().__init__()
        
        self.name = name
        self.module = module
    
    def forward(self, x):
        with timer(self.name):
            return self.module(x)

class ChannelSplit(nn.Module):
    def __init__(self, split):
        super().__init__()
        
        self.split = split
    
    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, C, H, self.split, W // self.split).permute(0, 1, 3, 2, 4).reshape(N, C*self.split, H, W//self.split)

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
        self.layer_id = -1

        #- attention predictor
        #-- mlp predictor
        self.performer_nb_features = int(
            self.attention_head_size * math.log(self.attention_head_size) / self.pconfig.performer_nb_factor
        )
        self.performer = FastAttention(
            dim_heads = self.attention_head_size,
            nb_features = self.performer_nb_features,
            causal=self.pconfig.causal,
            generalized_attention=self.pconfig.causal,
        )
        self.performer_proj_updater = ProjectionUpdater(
            self.performer,
            1000,
        )
        if os.environ.get("PERLIN_IGNORE_COSFORMER", "0") == "0" and self.pconfig.attention_predictor_backend == 'cosformer':
            from ..cosformer import CosformerAttention
            # cosformer not supported because of v dim does not supports custom
            self.cosformer = CosformerAttention(
                embed_dim=self.all_head_size,
                vdim=self.all_head_size*2,
                num_heads=self.num_attention_heads,
                has_outproj=False,
                causal=True,
            )
        if not self.pconfig.causal:
            performer_value_hidden_size = self.attention_head_size*3
        else:
            performer_value_hidden_size = self.attention_head_size*3
        self.register_buffer('attention_predictor_enc_head_embd', None)
        self.attention_predictor_enc_head_embd = torch.eye(self.num_attention_heads)
        self.attention_predictor_enc_per_layer = nn.Sequential(
            nn.Linear(performer_value_hidden_size * self.num_attention_heads, self.attention_head_size * 2 * self.num_attention_heads),
            nn.LayerNorm(self.attention_head_size * 2 * self.num_attention_heads),
            nn.GELU(),
        )
        self.attention_predictor_enc = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(performer_value_hidden_size, self.attention_head_size*2),
            # nn.Linear(performer_value_hidden_size + self.num_attention_heads, self.attention_head_size*2),
            nn.LayerNorm(self.attention_head_size*2),
            nn.GELU(),
        )
        N_H = self.num_attention_heads
        if not self.pconfig.causal:
            self.attention_predictor_dec_row_down_scale = 2
            self.attention_predictor_dec_row_splits = 4
            self.attention_predictor_dec_row_out_ch = (self.pconfig.attention_predictor_length // self.attention_predictor_dec_row_down_scale) * self.attention_predictor_dec_row_splits
            self.attention_predictor_dec_row = nn.Sequential(
                nn.Linear(self.attention_head_size*2, self.attention_predictor_dec_row_out_ch),
                ChannelSplit(self.attention_predictor_dec_row_splits),
            )
            self.attention_predictor_cnn = nn.Sequential(
                KeepRes(
                    nn.Conv2d(self.attention_predictor_dec_row_splits*N_H, 4*N_H, 3, padding=1, stride=(2, 1)),
                    # nn.Conv2d(4*N_H, 4*N_H, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(4*N_H, 4*N_H, 3, padding=1),
                    nn.ReLU(),
                    UpsampleFP32((2, 1), torch.float16),
                    nn.Conv2d(4*N_H, N_H, 3, padding=1),
                    output_width=self.pconfig.attention_predictor_length
                ),
            )
            # self.attention_predictor_dec_row = nn.Sequential(
            #     nn.Linear(self.attention_head_size*2, self.pconfig.attention_predictor_length),
            #     # ChannelSplit(self.attention_predictor_dec_row_splits),
            # )
            # self.attention_predictor_cnn = nn.Sequential(
            #     KeepRes(
            #         # NOTE if we use pixelshuffle outch should be 48
            #         CausalConv2d(12, 48, 3, padding=1, stride=2, causal=self.pconfig.causal),
            #         nn.ReLU(),
            #         ResBlock(48, causal=self.pconfig.causal),
            #         ResBlock(48, causal=self.pconfig.causal),
            #         UpsampleFP32(2, torch.float16),
            #         CausalConv2d(48, 12, 3, padding=1, causal=self.pconfig.causal),
            #     )
            # )
        else:
            is_causal = self.pconfig.causal
            inner_ch = 2
            self.attention_predictor_dec_row_down_scale = 4
            self.attention_predictor_dec_row_splits = inner_ch
            self.attention_predictor_dec_row_out_ch = (self.pconfig.attention_predictor_length // self.attention_predictor_dec_row_down_scale) * self.attention_predictor_dec_row_splits
            self.attention_predictor_dec_row = nn.Sequential(
                nn.Linear(self.attention_head_size*2, self.attention_predictor_dec_row_out_ch),
                ChannelSplit(self.attention_predictor_dec_row_splits),
            )
            self.attention_predictor_cnn = nn.Sequential(
                ModuleBenchmark('cnn.lnorm1', nn.LayerNorm(self.pconfig.attention_predictor_length // self.attention_predictor_dec_row_down_scale)),
                ModuleBenchmark('cnn.keepres', KeepRes(
                    # ModuleBenchmark('cnn.keepres.conv1', CausalConv2d(N_H, inner_ch*N_H, 5, padding=2, stride=(1, 4), causal=is_causal)),
                    # nn.ReLU(),
                    ModuleBenchmark('cnn.keepres.conv1', CausalConv2d(inner_ch*N_H, inner_ch*N_H, 3, padding=2, dilation=2, stride=(1, 1), causal=is_causal)),
                    nn.ReLU(),
                    ModuleBenchmark('cnn.keepres.conv2', CausalConv2d(inner_ch*N_H, inner_ch*N_H, 3, padding=2, dilation=2, stride=(1, 1), causal=is_causal)),
                    nn.ReLU(),
                    ModuleBenchmark('cnn.keepres.upsam', UpsampleFP32((1, 4), torch.float16)),
                    ModuleBenchmark('cnn.keepres.conv4', CausalConv2d(inner_ch*N_H, N_H, 1, padding=1, causal=is_causal)),
                    output_width=self.pconfig.attention_predictor_length
                )),
                # this prevent model explode within causal setting...
                ModuleBenchmark('cnn.lnorm2', nn.LayerNorm(self.pconfig.attention_predictor_length))
            )
            # self.attention_predictor_cnn = torch.compile(self.attention_predictor_cnn, mode='reduce-overhead')
        # self.attention_predictor_cnn = nn.Identity()
        self.attention_predictor_dec_scaler = nn.Sequential(
            nn.Linear(self.attention_head_size*2, 2),
        )
        
        #-- compressed predictor
        self.attention_predictor_comp_length = \
            self.pconfig.attention_predictor_comp_patch_count * self.pconfig.attention_predictor_comp_patch_size
        self.attention_predictor_comp_codebook = nn.Parameter(
            torch.randn((self.pconfig.attention_predictor_comp_book_size, self.pconfig.attention_predictor_comp_patch_size))
        )
        self.attention_predictor_comp_enc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(performer_value_hidden_size, self.attention_head_size*2),
            nn.LayerNorm(self.attention_head_size*2),
            nn.GELU(),
        )
        self.attention_predictor_comp_dec_row = nn.Sequential(
            nn.Linear(
                self.attention_head_size*2,
                self.pconfig.attention_predictor_comp_book_size * self.pconfig.attention_predictor_comp_patch_count
            ),
        )
        #-- TODO VQVAE
        
        #- output
        self.norm_performer = nn.LayerNorm(config.hidden_size)
        self.norm_partial = nn.LayerNorm(config.hidden_size)
        self.norm_random = nn.LayerNorm(config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        self.register_buffer('_v_eye', None, persistent=False)
        
        self.v_eye_learned = nn.Parameter(
            data=torch.rand((1, 1, self.attention_head_size, self.attention_head_size)),
            requires_grad=True
        )
        
        self.v_eye_learned_causal = nn.Parameter(
            data=torch.randn((1, 1, 2048, self.attention_head_size)),
            requires_grad=True
        )
    
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
        last_state: PerlinAttentionState = None,
    ):
        if context_layer_truth is not None and context_layer_truth.device != q.device:
            context_layer_truth = context_layer_truth.to(q.device, non_blocking=True)
            attention_scores_truth = attention_scores_truth.to(q.device, non_blocking=True)
        
        DUMMY_OUTPUT = PerlinAttentionOutput(
            loss=None,
            context_layer=None,
            partial_attention_probs=None,
            partial_attention_mask=None,
            estimated_attention_probs=None,
            estimated_attention_probs_m=None,
            dense_attention_probs=None,
            key_for_score=None,
            state=None,
        )
        
        use_cache = self.pconfig.use_cache
        
        if q.dtype in [torch.float16, torch.bfloat16]:
            # NOTE HJ even if we are in bfloat16, we have to use fp16 minimum because of F.interpolate
            FP_MIN = torch.finfo(torch.float16).min / 2
        elif q.dtype in [torch.float32]:
            FP_MIN = torch.finfo(torch.float32).min / 2
        else:
            raise Exception('unknown type')
        
        _, _, _, T_SRC = attention_mask.shape
        T_DST = T_SRC
        if self.pconfig.causal:
            if not use_cache:
                N, H, T_DST, T_SRC = attention_mask.shape
                assert T_DST == T_SRC
                assert H == 1
                causal_attention_mask = attention_mask
                attention_mask = attention_mask[:, :, :, :1].transpose(-1, -2)
            else:
                N, H, T_DST, T_SRC = attention_mask.shape
                _N, _H, _T_DST, _HID_Q = q.shape
                _N, _H, _T_SRC, _HID_K = k.shape
                assert k.shape[:-2] == v.shape[:-2]
                assert T_DST == _T_DST
                assert T_SRC == _T_SRC
                assert _HID_Q == _HID_K
                
                causal_attention_mask = attention_mask
                attention_mask = causal_attention_mask[:, :, -1:, :]
                
                assert attention_mask.shape == (1, 1, 1, T_SRC)
                assert causal_attention_mask.shape == (1, 1, T_DST, T_SRC)
        
        # return DUMMY_OUTPUT #0
        
        dst_attention_mask = attention_mask.transpose(-1, -2)
        if self.pconfig.causal:
            dst_attention_mask = causal_attention_mask[:,:,:,:1]
        
        not_padded = (attention_mask > -1).float().sum() == attention_mask.numel()
        
        if use_cache and last_state is None:
            last_state = PerlinAttentionState(self)
        if not use_cache:
            last_state = None
        
        # print('state', use_cache, strify(last_state), strify(q), strify(k), strify(v))
        
        raise_if_nan(q)
        raise_if_nan(k)
        raise_if_nan(v)
        
        zero_one_attention_mask = (attention_mask > -1).float()
        zero_one_attention_mask_cumsum = zero_one_attention_mask.cumsum(-1)
        zero_one_attention_mask_sum = zero_one_attention_mask.sum(-1)
        
        get_bench().register_temp_buffer('q', q)
        get_bench().register_temp_buffer('k', k)
        get_bench().register_temp_buffer('v', v)
        get_bench().register_temp_buffer('attention_mask', attention_mask)
        
        get_bench().register_temp_buffer('q', q)
        get_bench().register_temp_buffer('k', k)
        get_bench().register_temp_buffer('v', v)
        get_bench().register_temp_buffer('q_for_atten', q_for_atten)
        get_bench().register_temp_buffer('k_for_atten', k_for_atten)
        get_bench().register_temp_buffer('v_for_atten', v_for_atten)
        get_bench().register_temp_buffer('q_for_score', q_for_score)
        get_bench().register_temp_buffer('k_for_score', k_for_score)
        
        with timer("perlin"):
            # return DUMMY_OUTPUT #0
            
            N, H, T, HID = q.shape
            with timer("vmask"):
                # if not causal, we just use Eye matrix for V_identity
                if not self.pconfig.causal:
                    with timer("vmaks.eye"):
                        # E_N = min(T, HID)
                        E_N = HID
                        
                        if self._v_eye is None or self._v_eye.shape[-1] != E_N or self._v_eye.dtype != v.dtype:
                            v_for_atten_identity = torch.eye(
                                n=E_N,
                                dtype=v.dtype,
                                device=v.device,
                            )
                            
                            v_for_atten_identity = v_for_atten_identity.view(1, 1, E_N, E_N)
                            self._v_eye = v_for_atten_identity
                        else:
                            v_for_atten_identity = self._v_eye
                        
                        v_for_atten_identity = v_for_atten_identity.expand(v_for_atten.shape[:2] + (E_N, E_N))
                        get_bench().register_temp_buffer('v_for_atten_identity_bef_grid', v_for_atten_identity)

                    
                    with timer("vmask.grid"):
                        token_index_y = ((zero_one_attention_mask_cumsum - 1.0) / ((zero_one_attention_mask_sum - 1.0).view(N, 1, 1, 1) + 1e-8) * 2 - 1)\
                            .view(N, T, 1, 1)\
                            .expand(N, T, HID, 1)
                        token_index_x = (torch.arange(HID, device=q.device, dtype=torch.long) / (HID - 1) * 2 - 1).view(1, 1, HID, 1)
                        token_index_x = token_index_x.expand(N, T, HID, 1)
                        token_index = torch.cat([token_index_x, token_index_y], dim=-1)
                    
                    with timer("vmask.sample"):
                        v_for_atten_identity = grid_sample_bf16(
                            input=v_for_atten_identity, 
                            grid=token_index.to(v_for_atten_identity.dtype), 
                            mode='bilinear',
                            align_corners=True,
                        )
                        get_bench().register_temp_buffer('v_for_atten_identity_aft_grid', v_for_atten_identity)

                
                with timer("vmask.cat_fill"):
                    if not self.pconfig.causal:
                        v_for_atten = torch.cat([
                            v_for_atten_identity, 
                            v_for_atten
                        ], dim=-1)
                    else:
                        v_for_atten_pos_emb = self.v_eye_learned_causal[:,:,:T_SRC,:]
                        v_for_atten = torch.cat([
                            v_for_atten_pos_emb.expand(v_for_atten.shape),
                            v_for_atten
                        ], dim=-1)
                    
                    get_bench().register_temp_buffer('v_for_atten', v_for_atten)

                    if not not_padded:
                        v_for_atten.masked_fill_(dst_attention_mask < -1, 0)
                        v.masked_fill_(dst_attention_mask < -1, 0)
            
            # return DUMMY_OUTPUT #12
            
            with timer("performer"):
                if not self.benchmarking:
                    q_type = q_for_atten.dtype
                    if torch.get_autocast_gpu_dtype() in [torch.float16, torch.bfloat16]:
                        PRECISION_PERF = torch.float32 if torch.cuda.is_bf16_supported() else torch.float32
                    else:
                        PRECISION_PERF = torch.float32
                    with torch.autocast('cuda', PRECISION_PERF):
                        if self.pconfig.attention_predictor_backend == 'performer':
                            last_state, performer_context_layer = PerlinAttentionState.stateful_performer(
                                last_state,
                                "performer->performer_context_layer",
                                self.performer,
                                batch_to(q_for_atten, PRECISION_PERF), 
                                batch_to(k_for_atten, PRECISION_PERF), 
                                batch_to(v_for_atten, PRECISION_PERF),
                            )
                        elif self.pconfig.attention_predictor_backend == 'cosformer':
                            _q = batch_to(q_for_atten, PRECISION_PERF)
                            _k = batch_to(k_for_atten, PRECISION_PERF)
                            _v = batch_to(v_for_atten, PRECISION_PERF)
                            _N, _H, _T, _HID = _q.shape
                            _q = _q.permute(0, 2, 1, 3).reshape(_N, _T, _H*_HID).permute(1, 0, 2)
                            _N, _H, _T, _HID = _k.shape
                            _k = _k.permute(0, 2, 1, 3).reshape(_N, _T, _H*_HID).permute(1, 0, 2)
                            _N, _H, _T, _HID = _v.shape
                            _v = _v.permute(0, 2, 1, 3).reshape(_N, _T, _H*_HID).permute(1, 0, 2)
                            t = self.cosformer(_q, _k, _v)
                            print(t.shape)
                            performer_context_layer = t.reshape(_T, _N, _H, _HID).permute(1, 2, 0, 3)
                            # print('hello')
                        else:
                            raise Exception(self.pconfig.attention_predictor_backend)
                        
                    if q_type != performer_context_layer.dtype:
                        performer_context_layer = performer_context_layer.to(q_type)
                    assert performer_context_layer.shape[-2] == q.shape[-2], f"{performer_context_layer.shape} == {q.shape}, {v_for_atten.shape}"
                    # print('pcl', strify(performer_context_layer), strify(q), strify(k), strify(v))
                else:
                    # TODO: fix numerical stability...
                    performer_context_layer = self.performer(
                        q_for_atten, 
                        k_for_atten, 
                        v_for_atten
                    )
                    # q_type = q_for_atten.dtype
                    # if torch.get_autocast_gpu_dtype() in [torch.float16, torch.bfloat16]:
                    #     PRECISION_PERF = torch.float32 if torch.cuda.is_bf16_supported() else torch.float32
                    # else:
                    #     PRECISION_PERF = torch.float32
                    # with torch.autocast('cuda', PRECISION_PERF):
                    #     last_state, performer_context_layer = PerlinAttentionState.stateful_performer(
                    #         last_state,
                    #         "performer->performer_context_layer",
                    #         self.performer,
                    #         q_for_atten, 
                    #         k_for_atten, 
                    #         v_for_atten,
                    #     )
                    # if q_type != performer_context_layer.dtype:
                    #     performer_context_layer = performer_context_layer.to(q_type)
                    
                get_bench().register_temp_buffer('performer_context_layer', performer_context_layer)
                get_bench().register_temp_buffer('performer_context_layer>0', performer_context_layer>0)
                
            
            # return DUMMY_OUTPUT #119
            
            with timer("performer_value"):
                # NOTE May cut gradient from loss_sp, because loss_sp has sometimes negative effect to loss_model when approximation is sucks.
                if performer_context_layer.shape[-2] < v.shape[-2]:
                    performer_value = torch.cat([
                        performer_context_layer, 
                        v[...,-performer_context_layer.shape[-2]:,:]
                    ], dim=-1)#.detach()
                else:
                    performer_value = torch.cat([
                        performer_context_layer, 
                        v
                    ], dim=-1)#.detach()
                raise_if_nan(performer_value)
                get_bench().register_temp_buffer('performer_value', performer_value)
            
            # return DUMMY_OUTPUT #120
            
            # estimate attention scores
            with timer("predictor"):
                if self.pconfig.attention_predictor_method == 'mlp':
                    with timer("predictor.enc"):
                        raise_if_nan(performer_value)
                        # ENC_PER_LAYER = False
                        if self.pconfig.attention_predictor_enc_per_layer:
                            # print('hello')
                            _N, _H, _T, _D = performer_value.shape
                            t_enc_x = performer_value.permute(0, 2, 1, 3).reshape(_N, _T, _H*_D)
                            t_attention_predictor = self.attention_predictor_enc_per_layer(t_enc_x)
                            _HD = t_attention_predictor.shape[-1]
                            t_attention_predictor = t_attention_predictor.view(N, _T, _H, _HD // _H).permute(0, 2, 1, 3)
                            assert t_attention_predictor.shape[:3] == (_N, _H, _T)
                        else:
                            t_enc_x = performer_value
                            # _N, _H, _T, _D = performer_value.shape
                            # t_enc_x = torch.cat([
                            #     performer_value, 
                            #     self.attention_predictor_enc_head_embd.view(1, _H, 1, _H).expand(_N, _H, _T, _H)
                            # ], dim=-1)
                            t_attention_predictor = self.attention_predictor_enc(t_enc_x)
                    with timer("predictor.dec_row"):
                        raise_if_nan(t_attention_predictor)
                        estimated_attention_score = self.attention_predictor_dec_row(t_attention_predictor) # type: torch.Tensor
                        get_bench().register_temp_buffer('estimated_attention_score_dec_row', estimated_attention_score)
                        raise_if_nan(estimated_attention_score)
                    
                    with timer("predictor.cnn"):
                        last_state, estimated_attention_score = PerlinAttentionState.stateful_causal_cnn_op(
                            last_state,
                            "attention_predictor_cnn->estimated_attention_score",
                            self.attention_predictor_cnn,
                            estimated_attention_score,
                            q.shape[-2],
                        )
                        
                        # print('cnnd', strify(last_state), q.shape, strify(estimated_attention_score))
                        assert estimated_attention_score.shape[-2] == T_DST
                        # print('estimated_attention_score', strify(estimated_attention_score))
                elif self.pconfig.attention_predictor_method == 'comp':
                    assert not use_cache
                    warnings.warn('attention prediction method is compressed one.')
                    t_attention_predictor = self.attention_predictor_comp_enc(performer_value)
                    estimated_attention_score = self.attention_predictor_comp_dec_row(t_attention_predictor)
                    estimated_attention_score = estimated_attention_score\
                        .view(N, H, T, self.pconfig.attention_predictor_comp_patch_count, self.pconfig.attention_predictor_comp_book_size)
                    _, _, _, CODE_SEQ_LEN, BOOK_LEN = estimated_attention_score.shape
                    estimated_attention_score = softmax_bf16(estimated_attention_score, dim = -1)
                    estimated_attention_score = torch.matmul(
                        estimated_attention_score.view(-1, BOOK_LEN), 
                        self.attention_predictor_comp_codebook
                    )
                    estimated_attention_score = estimated_attention_score.view(N, H, T, -1)
                else:
                    raise Exception()
                get_bench().register_temp_buffer('t_attention_predictor', t_attention_predictor)
            
            # return DUMMY_OUTPUT #413
        
            # interpolate and convert to probability
            with timer("mask_softmax"):
                T_M = estimated_attention_score.shape[-1]
                estimated_attention_probs = softmax_bf16(estimated_attention_score, -1)
                assert estimated_attention_probs.shape[-2] == T_DST, f"{estimated_attention_probs.shape}, {T_DST}"
            
            # return DUMMY_OUTPUT #413
            
            estimated_attention_score_init = estimated_attention_score.clone().permute(0, 2,1,3).reshape(N, T, H*T_M)
            estimated_attention_probs_init = estimated_attention_probs.clone().permute(0, 2,1,3).reshape(N, T, H*T_M)
            get_bench().register_temp_buffer('estimated_attention_score_init', estimated_attention_score_init)
            get_bench().register_temp_buffer('estimated_attention_probs_init', estimated_attention_probs_init)
            
            # in layerwise, train perlin attention predictor
            # TODO set up resize_m_to_t
            def resize_from_m_to_t(x, masked_fill_value, target_width=None, output_dtype=None):
                from .ops import resize_from_m_to_t
                N, H, T1, T_M = x.shape
                if target_width is not None:
                    T2 = target_width
                else:
                    T2 = T1
                
                mask = attention_mask
                if self.pconfig.causal:
                    mask = causal_attention_mask
                
                return resize_from_m_to_t(
                    x=x,
                    masked_fill_value=masked_fill_value,
                    attention_mask=mask,
                    target_width=target_width,
                    training=self.training and self.pconfig.causal,
                    is_causal=self.pconfig.causal,
                )
            
            loss = 0
            estimated_attention_probs_resized = estimated_attention_score_resized = None
            if not self.benchmarking and not use_cache and attention_scores_truth is not None:
                N, H, T, T_M = estimated_attention_score.shape
                # for loss calculation
                # with torch.no_grad():
                estimated_attention_probs_resized = resize_from_m_to_t(estimated_attention_probs, masked_fill_value=0)
                # return DUMMY_OUTPUT #413
                estimated_attention_score_resized = resize_from_m_to_t(estimated_attention_score, masked_fill_value=FP_MIN, output_dtype=torch.float32)
                # return DUMMY_OUTPUT #601
                
                with torch.autocast('cuda', torch.float32):
                    raise_if_nan(estimated_attention_score_resized)
                    raise_if_nan(attention_scores_truth)
                    
                    if not self.pconfig.causal:
                        loss_kl_t = kl_div_attention(
                            F.log_softmax(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                            F.softmax(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                            attention_mask,
                        ) * 0.1
                        loss_mse_t = F.mse_loss(
                            softmax_bf16(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN), dim=-1), 
                            softmax_bf16(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1)
                        )
                    else:
                        # return DUMMY_OUTPUT #838
                        with torch.autocast('cuda', torch.float32):
                            _t_causal_mask = causal_attention_mask < -1
                            # return DUMMY_OUTPUT #601
                            _input = F.log_softmax(estimated_attention_score_resized.masked_fill_(_t_causal_mask, FP_MIN), dim=-1, dtype=torch.float32).view(-1, estimated_attention_score_resized.shape[-1])
                            # return DUMMY_OUTPUT #751
                            _target = F.softmax(attention_scores_truth.masked_fill_(_t_causal_mask, FP_MIN), dim=-1, dtype=torch.float32).view(-1, estimated_attention_score_resized.shape[-1])
                            # return DUMMY_OUTPUT #942
                            loss_kl_t = F.kl_div(
                                _input,
                                _target,
                                reduction='batchmean',
                            ) * 0.1
                            del _input
                            # return DUMMY_OUTPUT #1518
                            loss_mse_t = F.mse_loss(
                                F.softmax(estimated_attention_score_resized, dim=-1, dtype=torch.float32).view(-1, estimated_attention_score_resized.shape[-1]),
                                _target,
                            )
                            del _target
                    
                    raise_if_nan(loss_kl_t)
                    raise_if_nan(loss_mse_t)
                    loss += loss_kl_t + loss_mse_t
                    raise_if_nan(loss)
                
                # JINA_VIZ_COLSEL 1 2
                #print('attention_probs_truth', lambda: F.softmax(attention_scores_truth, dim=-1) * (attention_mask.transpose(-1, -2) > -1))
                #print('attention_probs_truth_m', lambda: F.softmax(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1) * (attention_mask.transpose(-1, -2) > -1))
                #print('estimated_attention_probs_resized', estimated_attention_probs_resized)
                #print('estimated_attention_score_resized', estimated_attention_score_resized)
                get_bench().register_temp_buffer('attention_scores_truth', attention_scores_truth.clone().permute(0,2,1,3).reshape(N, T, H*T))
                get_bench().register_temp_buffer('attention_probs_truth', None, lazy=lambda: (F.softmax(attention_scores_truth, dim=-1) * (attention_mask.transpose(-1, -2) > -1)).clone().permute(0,2,1,3).reshape(N, T, H*T))
                get_bench().register_temp_buffer('attention_probs_truth_m', None, lazy=lambda: (F.softmax(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1) * (attention_mask.clone().transpose(-1, -2) > -1)).permute(0,2,1,3).reshape(N, T, H*T_M))
                get_bench().register_temp_buffer('estimated_attention_probs_resized', estimated_attention_probs_resized.clone().permute(0,2,1,3).reshape(N, T, H*T))
                get_bench().register_temp_buffer('estimated_attention_score_resized', estimated_attention_score_resized.clone().permute(0,2,1,3).reshape(N, T, H*T))
                    
                get_bench().register_temp_buffer('attention_probs_truth', None, lazy=lambda: F.softmax(attention_scores_truth, dim=-1) * (dst_attention_mask > -1))
                # if not self.pconfig.causal:
                #     get_bench().register_temp_buffer('attention_probs_truth_m', None, lazy=lambda: F.softmax(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1) * (dst_attention_mask > -1))
                get_bench().register_temp_buffer('estimated_attention_probs_resized', estimated_attention_probs_resized)
                get_bench().register_temp_buffer('estimated_attention_score_resized', estimated_attention_score_resized)
            
            # return DUMMY_OUTPUT #1518
            
            with timer("mask"):
                masked_estimated_attention_probs = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1))
                # TODO: perform this with states
                
                if not not_padded:
                    estimated_attention_probs = estimated_attention_probs * (dst_attention_mask > -1)
                get_bench().register_temp_buffer('masked_estimated_attention_probs', masked_estimated_attention_probs.clone().permute(0,2,1,3).reshape(N, T, H*T_M))
                
                N, H, T, T_M = estimated_attention_probs.shape
                assert T == T_DST, f"{T}=={T_DST}, {estimated_attention_probs.shape} {not_padded}"
                token_length = (attention_mask > -1).long().sum(-1).view(N, -1)
                top_k = min(max(int(round(self.pconfig.k * (T_M / torch.min(token_length).item()))), 1), T_M)
                # k_m = (torch.floor(self.pconfig.k * T_M / token_length)) # NOTE changeable
                k_flatten = self.pconfig.k_flatten
                perlin_col_select = self.pconfig.colsel
                warnings.warn(f"k_flatten {k_flatten}")

                if not k_flatten:
                    assert perlin_col_select == False
                    warnings.warn(f"perlin_col_select {perlin_col_select}")
                    with timer("mask.topk"):
                        _, indices = torch.topk(
                            estimated_attention_probs, # estimation gradient is cut here
                            k=top_k, 
                            dim=-1, 
                            sorted=True,
                        ) # [N, H, T, topk]
                    with timer("mask.empty"):
                        partial_attention_mask = torch.empty(
                            (N, H, T, T_M),
                            dtype=q_for_score.dtype,
                            device=q_for_score.device,
                        ) # [N, H, T, T_M]
                    with timer("mask.fill"):
                        partial_attention_mask.fill_(FP_MIN)
                    with timer("mask.scatter"):
                        partial_attention_mask.scatter_(dim=-1, index=indices, value=0)
                else:
                    k_flatten_dim = self.pconfig.k_flatten_dim
                    assert k_flatten_dim in ['batch', 'head', 'causal_batch']
                    warnings.warn(f"k_flatten_dim {k_flatten_dim}")
                    warnings.warn(f"perlin_col_select {perlin_col_select}")

                    top_k_elems = None
                    per_item_top_k = None 
                    k_flatten_dim = self.pconfig.k_flatten_dim
                    assert k_flatten_dim in ['head', 'batch', 'causal_batch']
                    with timer("mask.view"):
                        masked_estimated_attention_probs = (estimated_attention_probs * (dst_attention_mask > -1))
                        
                        get_bench().register_temp_buffer('masked_estimated_attention_probs', masked_estimated_attention_probs)
                        
                        if not self.pconfig.causal:
                            token_length = (attention_mask > -1).long().sum(-1).view(N, -1)
                        else:
                            causal_token_length = (causal_attention_mask > -1).long().sum(-1).view(1, 1, T_DST, 1)
                        
                        if k_flatten_dim == 'batch':
                            assert not self.pconfig.causal

                            if not perlin_col_select:
                                t = masked_estimated_attention_probs.view(N, H*T*T_M)
                            per_t_in_item_top_k = H * torch.round(self.pconfig.k * T_M / token_length)
                        # top_k_elems = top_k*T*H
                                
                        elif k_flatten_dim == 'head':
                            assert not self.pconfig.causal
                            if not perlin_col_select:
                                t = masked_estimated_attention_probs.view(N, H, T*T_M)
                            per_t_in_item_top_k = torch.round(self.pconfig.k * T_M / token_length)
                            # top_k_elems = top_k*T
                        elif k_flatten_dim == 'causal_batch':
                            t = masked_estimated_attention_probs.transpose(1, 2).reshape(N, T, H*T_M)
                            # top_k_elems = top_k*H
                            # per_item_top_k = (H * self.pconfig.k)
                            if not self.pconfig.causal:
                                per_item_top_k = (H * torch.round(self.pconfig.k * T_M / token_length)).view(N, 1, 1)
                            else:
                                # NOTE consider causal token length
                                per_item_top_k = torch.clamp((H * torch.round(self.pconfig.k * T_M / causal_token_length.squeeze(0))).view(1, T_DST, 1), 1, H*T_M)
                        else: raise Exception()

                    if perlin_col_select: # TODO change timer, self.benchmark case
                        assert k_flatten_dim in ['batch', 'head'] # TODO add causal_batch
                        col_select_method = self.pconfig.colsel_method
                        mask_in_probs = self.pconfig.colsel_mask_in_probs
                        assert col_select_method in ["mean_values", "sum_values", "sum_mask",]
                        warnings.warn(f"col_select_method {col_select_method}")
                        warnings.warn(f"mask_in_probs {mask_in_probs}")

                        # TODO currently implemented with the thought that it gets thicker, need to consider case of being thinner
                        # TODO add test_set for checking whether per_item_col_thickness is right
                        # input : per_item_col_thickness, interpolate rate(T Ti T_M), considering interpolation method
                        per_item_col_thickness = torch.min(torch.max(torch.round(1* T_M/token_length), torch.tensor([1]).to(token_length.device)), torch.tensor([T_M]).to(token_length.device))
                        get_bench().register_temp_buffer('per_item_col_thickness', per_item_col_thickness) # N, 1
                        
                        # TODO CHECK # [N, 1]
                        # NOTE t shape differs with topk TODO check this
                        col_t = masked_estimated_attention_probs.permute(0, 2, 1, 3).reshape(N, T, H*T_M) if k_flatten_dim=='batch' else masked_estimated_attention_probs

                        # TODO learnable mean_threshold?
                        if k_flatten_dim == 'batch':
                            assert not self.pconfig.causal
                            if col_select_method =='mean_values':
                                mean_threshold = 1/T_M
                                col_mean_in_flatten = col_t.sum(dim=-2)/token_length # N, H*T_M
                                col_result_mask = col_mean_in_flatten >= mean_threshold # N, H*T_M
                                col_result_mask_cnt = col_result_mask.sum(dim=-1, keepdim=True) # N, 1
                                
                                get_bench().register_temp_buffer('col_mean_in_flatten', col_mean_in_flatten)
                                get_bench().register_temp_buffer('col_result_mask', col_result_mask)
                                get_bench().register_temp_buffer('col_result_mask_cnt', col_result_mask_cnt)

                                # TODO result of per_item_top_k_col_real could be bigger than per_t_in_item_top_k (unless per_item_col_thickness is very accurate)
                                # 1. check before going to topk, scatter fill in the low places (***)
                                # 2. handle this in inflated_per_item_col : considering resizing method (****)

                                # NOTE it seems reasonable to prioritize colselect than elementwise topk, in mean_values case, 
                                # inflated_per_item_col should not be generous, or mean_threshold should not be generous
                                # colsel_per_head_cnt_limit might fix this / or use upper threshold(than 1/T_M) in col_result_mask

                                # 1. by colsel_per_head_threshold (colsel_per_head_cnt_limit)
                                
                                # NOTE (not that critical, col_result_mask_cnt will handle this later on) colsel might leak error in the few initial training state 
                                # 1. lower effect of context layer loss
                                
                                # TODO learnable colsel_per_head_cnt_limit?; currently ~ hardcoded
                                if self.pconfig.colsel_per_head_cnt_limit > -1:
                                    assert self.pconfig.colsel_per_head_cnt_limit <= self.pconfig.k
                                    per_item_col_cnt_limit = torch.floor(self.pconfig.colsel_per_head_cnt_limit*self.num_attention_heads * T_M / token_length)
                                    assert (per_item_col_cnt_limit > per_t_in_item_top_k).sum().item() == 0
                                    warnings.warn(f'colsel_per_head_cnt_limit {self.pconfig.colsel_per_head_cnt_limit}')
                                    # NOTE col makes thing same for each T; inflated_per_item_col based on H*T_M or T_M
                                    # N, 1
                                    inflated_per_item_col = torch.min(torch.min(col_result_mask_cnt, torch.min(per_t_in_item_top_k*per_item_col_thickness, torch.tensor(col_t.shape[-1]))), 
                                                                        per_item_col_cnt_limit*per_item_col_thickness) # all thick columns are already included in col_result_mask_cnt
                                else:
                                    # per_t_in_item_top_k : in H*T_M or T_M
                                    inflated_per_item_col = torch.min(col_result_mask_cnt, torch.min(per_t_in_item_top_k*per_item_col_thickness, torch.tensor(col_t.shape[-1])))
                                
                                get_bench().register_temp_buffer('inflated_per_item_col', inflated_per_item_col) # N, 1
                                
                                inflated_top_k_elems_col = int(max(inflated_per_item_col))

                                def colsel(top_k_elems_col: int, per_item_col: torch.Tensor):
                                    with timer("mask.topk"):
                                        _, indices = torch.topk(
                                            input=col_mean_in_flatten, # N, H*T_M or N, H, T_M
                                            k=top_k_elems_col, 
                                            dim=-1, 
                                            sorted=True #sorted true is important
                                        ) # N, top_k_elems_col or N, H, top_k_elems_col

                                    with timer("mask.empty"):
                                        partial_attention_mask_col = torch.empty(
                                            col_t.shape, # N, T, H*T_M or N, H, T, T_M
                                            dtype=torch.long, 
                                            device=attention_mask.device,
                                        )
                                    with timer("mask.fill"):
                                        partial_attention_mask_col.fill_(col_t.shape[-1])
                                    with timer("mask.scatter"):
                                        partial_attention_mask_col.scatter_( # N, T, H*T_M or N, H, T, T_M
                                            dim=-1,
                                            index=indices.view((N, 1, top_k_elems_col) if k_flatten_dim=="batch"
                                                                else (N, H, 1, top_k_elems_col))\
                                                                    .expand((N, T, top_k_elems_col) if k_flatten_dim=="batch"
                                                                            else (N, H, T, top_k_elems_col)), # N, top_k_elems_col or N, H, top_k_elems_col
                                            src=torch.arange(
                                                top_k_elems_col, 
                                                dtype=torch.long,
                                                device=attention_mask.device, 
                                            )\
                                                .view((1, 1, -1) if col_t.ndim == 3 else (1, 1, 1, -1))\
                                                .expand((N, T, top_k_elems_col) if k_flatten_dim=="batch"
                                                        else (N, H, T, top_k_elems_col))
                                        )
                                        
                                    with timer("mask.masked_fill"):
                                        col_t_alive_mask = partial_attention_mask_col < per_item_col.view(N, 1, 1).expand(N, T, H*T_M) # N, T, H*T_M or N, H, T, T_M
                                        col_t_alive_mask_for_viz = col_t_alive_mask.clone()
                                        
                                        if k_flatten_dim == "batch":
                                            col_t_alive_mask = col_t_alive_mask.view(N, T, H, T_M).permute(0,2,1,3).float()
                                        elif k_flatten_dim == "head":
                                            col_t_alive_mask = col_t_alive_mask.view(N, H, T, T_M).float()
                                        else:
                                            raise Exception(f"k_flatten_dim {k_flatten_dim}")
                                        # view
                                        if not self.benchmarking:
                                            # TODO recheck resize_from_m_to_t, t_to_m, whether it works well
                                            col_t_alive_mask_t = resize_from_m_to_t(col_t_alive_mask, masked_fill_value=0)
                                            col_t_alive_mask_t_for_viz = col_t_alive_mask_t.clone().view(N, H, T, T).permute(0,2,1,3).reshape(N, T, H*T)
                                            col_t_alive_mask = resize_from_t_to_m(col_t_alive_mask_t, T_M)
                                            col_t_alive_mask_m_for_viz = col_t_alive_mask.clone().view(N, H, T, T_M).permute(0,2,1,3).reshape(N, T, H*T_M)

                                        else:
                                            # TODO check how this works!!!, TODO check per_item_col_thickness
                                            col_t_alive_mask = col_t_alive_mask.reshape(N*H, T, T_M).to_sparse_coo()
                                            col_t_alive_mask_t = resize_width(col_t_alive_mask, T/T_M)
                                            col_t_alive_mask_t_for_viz = col_t_alive_mask_t.clone().to_dense().view(N, H, T, T).permute(0,2,1,3).reshape(N, T, H*T)

                                            col_t_alive_mask = resize_width(col_t_alive_mask_t, T_M/T).to_dense().view(N, H, T, T_M)
                                            col_t_alive_mask_m_for_viz = col_t_alive_mask.clone().permute(0,2,1,3).reshape(N, T, H*T_M)

                                    assert col_t_alive_mask.shape == (N, H, T, T_M)
                                    col_t_alive_mask = col_t_alive_mask.permute(0, 2, 1, 3).reshape(N, T, H*T_M)
                                    sum_last_dim = col_t_alive_mask.sum(dim=-1) # N, T
                                    assert torch.all(sum_last_dim==sum_last_dim[:,0:1], dim=-1).all() # N, T has all same value in -1 dim
                                    per_item_col_real = sum_last_dim[:,0:1] # N, 1
                                    # warnings.warn(f"per_t_in_item_top_k {per_t_in_item_top_k}")
                                    # warnings.warn(f"per_item_col_real {per_item_col_real}")
                                    # print('col real', per_item_col_real)
                                    # print('col limit', per_item_col_cnt_limit)
                                    # print('t_in_topk', per_t_in_item_top_k)
                                    if self.pconfig.colsel_per_head_cnt_limit > -1:
                                        per_item_condition_not_satisfied = per_item_col_real > torch.min(per_t_in_item_top_k, per_item_col_cnt_limit)
                                    else:
                                        per_item_condition_not_satisfied = per_item_col_real > per_t_in_item_top_k
                                    return col_t_alive_mask, per_item_col_real, per_item_condition_not_satisfied,\
                                            col_t_alive_mask_for_viz, col_t_alive_mask_t_for_viz, col_t_alive_mask_m_for_viz

                                # TODO test script: test_perlin_col, and compare this with test_perlin_topk, final per_t_in_item_top_k
                                # TODO test script: test_perlin_col, per_t_in_item_top_k per layer - check if it's performing reasonable decision
                                col_t_alive_mask, per_item_col_real_1, per_item_condition_not_satisfied_1,\
                                col_t_alive_mask_for_viz_1, col_t_alive_mask_t_for_viz_1, col_t_alive_mask_m_for_viz_1= colsel(inflated_top_k_elems_col, inflated_per_item_col)
                                colsel_perform_cnt = 0 # for viz
                                get_bench().register_temp_buffer('per_item_col_real_1', per_item_col_real_1)
                                get_bench().register_temp_buffer('col_t_alive_mask_bef_inter_1', col_t_alive_mask_for_viz_1)
                                get_bench().register_temp_buffer('col_t_alive_mask_t_1', col_t_alive_mask_t_for_viz_1)
                                get_bench().register_temp_buffer('col_t_alive_mask_m_1', col_t_alive_mask_m_for_viz_1)
                                get_bench().register_temp_buffer('per_item_condition_not_satisfied_1', per_item_condition_not_satisfied_1)

                                # NOTE this happens especially in the initial training state
                                # TODO same process repeated twice -> doubled latency; any better way?
                                if int(per_item_condition_not_satisfied_1.sum()):
                                    per_item_col_real_1_for2 = per_item_col_real_1.clone()
                                    col_t_alive_mask_for_viz_1_for2 = col_t_alive_mask_for_viz_1.clone()
                                    col_t_alive_mask_t_for_viz_1_for2 = col_t_alive_mask_t_for_viz_1.clone()
                                    col_t_alive_mask_m_for_viz_1_for2 = col_t_alive_mask_m_for_viz_1.clone()
                                    per_item_condition_not_satisfied_1_for2 = per_item_condition_not_satisfied_1.clone()

                                    if self.pconfig.colsel_per_head_cnt_limit > -1:
                                        per_item_new_col = torch.min(per_item_col_real_1, torch.min(per_t_in_item_top_k, per_item_col_cnt_limit))
                                    else:
                                        per_item_new_col = torch.min(per_item_col_real_1, per_t_in_item_top_k)
                                    top_k_elems_new_col = int(max(per_item_new_col))
                                    # TODO don't do colsel twice, but update col_t_alive_mask with top_k_elems_new_col
                                    col_t_alive_mask, per_item_col_real_2, per_item_condition_not_satisfied_2,\
                                    col_t_alive_mask_for_viz_2, col_t_alive_mask_t_for_viz_2, col_t_alive_mask_m_for_viz_2= colsel(top_k_elems_new_col, per_item_new_col)
                                    
                                    colsel_perform_cnt += 1

                                    get_bench().register_temp_buffer('layer_id', self.layer_id)
                                    # NOTE I think this is not necessary now; once using layer_id
                                    get_bench().register_temp_buffer('per_item_col_real_1_for2', per_item_col_real_1_for2)
                                    get_bench().register_temp_buffer('col_t_alive_mask_bef_inter_1_for2', col_t_alive_mask_for_viz_1_for2)
                                    get_bench().register_temp_buffer('col_t_alive_mask_t_1_for2', col_t_alive_mask_t_for_viz_1_for2)
                                    get_bench().register_temp_buffer('col_t_alive_mask_m_1_for2', col_t_alive_mask_m_for_viz_1_for2)
                                    get_bench().register_temp_buffer('per_item_condition_not_satisfied_1_for2', per_item_condition_not_satisfied_1_for2)
                                
                                    get_bench().register_temp_buffer('per_item_col_real_2', per_item_col_real_2)
                                    get_bench().register_temp_buffer('col_t_alive_mask_bef_inter_2', col_t_alive_mask_for_viz_2)
                                    get_bench().register_temp_buffer('col_t_alive_mask_t_2', col_t_alive_mask_t_for_viz_2)
                                    get_bench().register_temp_buffer('col_t_alive_mask_m_2', col_t_alive_mask_m_for_viz_2)
                                    get_bench().register_temp_buffer('per_item_condition_not_satisfied_2', per_item_condition_not_satisfied_2)
                                    
                                    per_t_in_item_top_k = per_t_in_item_top_k-per_item_col_real_2
                                    if int(per_item_condition_not_satisfied_2.sum()): # some items exceeds required k_m
                                        # TODO better way to handle that item? - handle resize fun or manage per_t_in_item_top_k?
                                        # handle inflated_per_item_col? thickness?
                                        # NOTE it would get better later on tho..
                                        # NOTE or just select topk with per_t_in_item_top_k? : but interpolation might delete out some values
                                        # warnings.warn(f'colsel_m exceeds k_m :\n {per_t_in_item_top_k}')
                                        # fill 0 in indices that exceed required k_m
                                        per_t_in_item_top_k.masked_fill_(per_t_in_item_top_k<0, 0.0) # NOTE not satisfied bc per_head case; skipping this slight affect (due to interpolation)
                                        # per_t_in_item_top_k = per_t_in_item_top_k * (1-per_item_condition_not_satisfied_2.float()) # TODO check averge k + is using float okay?
                                    per_t_in_item_top_k_for2 = per_t_in_item_top_k.clone()
                                    get_bench().register_temp_buffer('per_t_in_item_top_k_for2', per_t_in_item_top_k_for2)
                                else:
                                    per_t_in_item_top_k = per_t_in_item_top_k-per_item_col_real_1
                                
                                get_bench().register_temp_buffer('colsel_perform_cnt', colsel_perform_cnt)

                                col_t_alive_mask_final = col_t_alive_mask.clone()
                                get_bench().register_temp_buffer('col_t_alive_mask_final', col_t_alive_mask_final)
                                col_t_before_masked = col_t.clone()
                                get_bench().register_temp_buffer('col_t_before_masked', col_t_before_masked)
                                # update score or probs
                                if mask_in_probs: # we're going to ignore the selected col during topk
                                    col_t.masked_fill_( # N, T, H*T_M or N, H, T, T_M
                                        mask = col_t_alive_mask > 0,
                                        value = 0
                                    )
                                    col_t_after_masked = col_t.clone()
                                    get_bench().register_temp_buffer('col_t_after_masked', col_t_after_masked)
                                    t = col_t.view(N, T, H, T_M).permute(0, 2, 1, 3).reshape(N, H*T*T_M)
                                else:
                                    # NOTE estimated_attention_score & probs are modified
                                    # TODO this could be problematic when there exists a row that contains all FP_MIN after masking
                                    # TODO currently used same name but using different name is better?
                                    estimated_attention_score = estimated_attention_score.permute(0, 2, 1, 3).reshape(N, T, H*T_M)\
                                        .masked_fill( # N, T, H*T_M
                                        mask = col_t_alive_mask > 0,
                                        value = FP_MIN
                                    ).view(N, T, H, T_M).permute(0, 2, 1, 3)
                                    raise_if_nan(estimated_attention_score)
                                    estimated_attention_probs = softmax_bf16(estimated_attention_score, -1)
                                    raise_if_nan(estimated_attention_probs)
                                    masked_estimated_attention_probs = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1))
                                    raise_if_nan(masked_estimated_attention_probs)
                                    t = masked_estimated_attention_probs.view(N, H*T*T_M)
                                    estimated_attention_score_after_mp0 = estimated_attention_probs.clone().permute(0,2,1,3).reshape(N, T, H*T_M)
                                    get_bench().register_temp_buffer('estimated_attention_score_after_mp0', estimated_attention_score_after_mp0)
                                    col_t_after_masked = masked_estimated_attention_probs.clone().permute(0,2,1,3).reshape(N, T, H*T_M)
                                    get_bench().register_temp_buffer('col_t_after_masked', col_t_after_masked)

                                    raise_if_nan(t)

                            elif col_select_method == "sum_mask":
                                raise Exception(f"col_select_method {col_select_method} not provided")
                                return
                                # mask all values larger than 1/T_M
                                col_select_mask = col_sel_estimated_attention_probs >= (1/T_M)
                                # sum these masked 1
                                for i in range(col_result_width):
                                    sum_value = col_select_mask[:,:,i:i+col_thickness].reshape(N, -1).sum(dim=-1) # N
                                    col_result[:,i] = sum_value
                            else:
                                raise Exception(f"col_select_method {col_select_method}")
                            
                        elif k_flatten_dim == "head":
                            # JIN TODO
                            raise Exception(f"colsel k_flatten_dim {k_flatten_dim} not provided yet")

                        elif k_flatten_dim == "causal_batch":
                            # JIN TODO
                            raise Exception(f"colsel k_flatten_dim {k_flatten_dim} not provided yet")
                        
                        else:
                            raise Exception(f"k_flatten_dim {k_flatten_dim}")
                    
                    # NOTE to prevent 0 top-k when large T and small T_m
                    per_item_top_k = token_length * per_t_in_item_top_k
                    per_item_top_k = torch.clamp_min(per_item_top_k, 1)

                    top_k_elems = min(int(math.ceil(torch.max(per_item_top_k).item())), t.shape[-1])
                    get_bench().register_temp_buffer('per_item_top_k', per_item_top_k)
                    get_bench().register_temp_buffer('per_t_in_item_top_k', per_t_in_item_top_k)
                    get_bench().register_temp_buffer('top_k_elems', None, lazy=lambda: torch.tensor(top_k_elems, dtype=torch.float64))

                with timer("mask.topk"):
                    if top_k_elems < t.shape[-1] * 0.9:
                        _, indices = torch.topk(
                            input=t,
                            k=top_k_elems, 
                            dim=-1, 
                            sorted=True #sorted true is important
                        )
                    else:
                        _, indices = torch.sort(
                            t,
                            dim=-1,
                            descending=True,
                            stable=False,
                        )
                        indices = indices[...,:top_k_elems]
                    get_bench().register_temp_buffer('topk_indices', indices.double())
                with timer("mask.empty"):
                    partial_attention_mask = torch.empty(
                        t.shape, 
                        dtype=torch.long, 
                        device=attention_mask.device,
                    )
                with timer("mask.fill"):
                    partial_attention_mask.fill_(t.shape[-1])
                with timer("mask.scatter"):
                    partial_attention_mask.scatter_(
                        dim=-1,
                        index=indices,
                        src=torch.arange(
                            top_k_elems, 
                            dtype=torch.long,
                            device=attention_mask.device, 
                        )\
                            .view((1, -1) if t.ndim == 2 else (1, 1, -1))\
                            .expand(indices.shape)
                    )
                    # #print(partial_attention_mask[0].view(H, T, T_M)[0])
                with timer("mask.masked_fill"):
                    if not self.benchmarking:
                        if k_flatten_dim=='batch': # TODO check for causal_batch case
                            t_dead_mask = partial_attention_mask >= per_item_top_k
                            if perlin_col_select:
                                t_dead_mask_before_colsel = t_dead_mask.float().view(N, H, T, T_M).permute(0, 2, 1, 3).contiguous().view(N, T, H*T_M)
                                get_bench().register_temp_buffer('t_dead_mask_before_colsel', None, lambda: t_dead_mask_before_colsel)
                            else:
                                get_bench().register_temp_buffer('t_dead_mask', None, lambda: t_dead_mask.float().view(N, H, T, T_M).permute(0, 2, 1, 3).contiguous().view(N, T, H*T_M)) # N, H*T*T_M
                        elif k_flatten_dim == 'head':
                            t_dead_mask = partial_attention_mask >= ((per_item_top_k).view(N, 1, 1).expand_as(partial_attention_mask)) # N, H, T*T_M
                            if perlin_col_select:
                                t_dead_mask_before_colsel = t_dead_mask.clone().float().view(N, H, T, T_M)
                                get_bench().register_temp_buffer('t_dead_mask_before_colsel', None, lambda: t_dead_mask_before_colsel) # want to see full view?
                            get_bench().register_temp_buffer('t_dead_mask', None, lambda: t_dead_mask.float().view(N, H, T, T_M))
                        elif k_flatten_dim == 'causal_batch':
                            raise Exception(f"k_flatten_dim {k_flatten_dim}")
                        else:
                            raise Exception(f"k_flatten_dim {k_flatten_dim}")
                        if perlin_col_select: # TODO check for head, causal batch
                            col_t_alive_mask = col_t_alive_mask.view(N, T, H, T_M).permute(0,2,1,3).reshape(N, H*T*T_M)
                            t_dead_mask.masked_fill_(mask=col_t_alive_mask > 0, value=0)
                            t_dead_mask_after_colsel = t_dead_mask.clone().float().view(N, H, T, T_M).permute(0, 2, 1, 3).contiguous().view(N, T, H*T_M)
                            get_bench().register_temp_buffer('t_dead_mask_after_colsel', None, lambda: t_dead_mask_after_colsel)
                        partial_attention_mask = t_dead_mask.to(q.dtype) * FP_MIN
                    else:
                        t_alive_mask = partial_attention_mask < per_item_top_k
                        get_bench().register_temp_buffer('t_alive_mask_before_colsel', None, lambda: t_alive_mask.clone().float().view(N, H, T, T_M).permute(0, 2, 1, 3).contiguous().view(N, T, H*T_M))
                        if k_flatten_dim=='batch':
                            if perlin_col_select:
                                col_t_alive_mask = col_t_alive_mask.view(N, T, H, T_M).permute(0,2,1,3).reshape(N, H*T*T_M)
                                t_alive_mask.masked_fill_(mask=col_t_alive_mask > 0, value=1)
                                get_bench().register_temp_buffer('t_alive_mask_after_colsel', None, lambda: t_alive_mask.clone().float().view(N, H, T, T_M).permute(0, 2, 1, 3).contiguous().view(N, T, H*T_M))
                        else:
                            raise Exception(f"k_flatten_dim {k_flatten_dim} not provided yet")
                        partial_attention_mask = t_alive_mask.float()

                if k_flatten_dim == 'causal_batch':
                    partial_attention_mask = partial_attention_mask.view(N, T, H, T_M).transpose(1, 2)
                    partial_attention_mask.masked_fill_(
                        mask=dst_attention_mask < -1,
                        value=FP_MIN
                    )
                elif k_flatten_dim in ['batch', 'head']:
                    pass
                else: raise Exception()
                partial_attention_mask = partial_attention_mask.view(N, H, T, T_M)
                if not self.benchmarking: # NOTE partial_attention_mask : in [0, FP_MIN]
                    partial_attention_mask.masked_fill_(
                        mask=attention_mask.transpose(-1, -2) < -1,
                        value=FP_MIN
                    ) # NOTE note that colsel fills 0 till T, not till token_length
                else: # NOTE partial_attention_mask : in [0, 1]
                    partial_attention_mask.masked_fill_(
                        mask=attention_mask.transpose(-1, -2) < -1,
                        value=0.0
                    ) # NOTE note that colsel fills 0 till T, not till token_length
            assert partial_attention_mask.shape ==(N, H, T, T_M)

            # TODO check what's wrong
            partial_attention_mask_before_interp = partial_attention_mask.clone().permute(0,2,1,3).reshape(N, T, H*T_M)
            get_bench().register_temp_buffer('partial_attention_mask_before_interp', partial_attention_mask_before_interp)

            # get_bench().register_temp_buffer('partial_attention_mask_before_interp', partial_attention_mask.clone())

            with timer("interp"):
                # NOTE: partial attention mask should be filled with 0 and -inf only.
                raise_if_nan(partial_attention_mask)
                if not self.benchmarking:
                    with timer("interp.resize"):
                        # TODO Fix this function to return COO tensor
                        # print('resize', strify(partial_attention_mask))
                        partial_attention_mask = resize_from_m_to_t(partial_attention_mask, FP_MIN, target_width=T_SRC)
                        if self.pconfig.causal:
                            partial_attention_mask.masked_fill_(causal_attention_mask < -1, FP_MIN)
                else:
                    if not self.pconfig.causal:
                        N, H, T, T_M = partial_attention_mask.shape
                        
                        # original
                        # partial_attention_mask_original = resize_from_m_to_t(partial_attention_mask, FP_MIN if not self.benchmarking else 0).view(N*H, T, T).to_sparse_coo()
                        
                        # optimized
                        partial_attention_mask_bef_coo = partial_attention_mask.clone().permute(0,2,1,3).reshape(N, T, H*T_M)
                        get_bench().register_temp_buffer('partial_attention_mask_bef_coo_intp', partial_attention_mask_bef_coo)
                        partial_attention_mask = partial_attention_mask.reshape(N*H, T, T_M).to_sparse_coo()
                        partial_attention_mask = resize_width(partial_attention_mask, T/T_M)
                        partial_attention_mask_aft_coo = partial_attention_mask.clone().to_dense().reshape(N, H, T, T).permute(0,2,1,3).reshape(N, T, H*T)
                        get_bench().register_temp_buffer('partial_attention_mask_aft_coo_intp', partial_attention_mask_aft_coo)
                        SPARSITY_TYPE = 'flat_csr'
                        if SPARSITY_TYPE == 'flat_csr':
                            with timer("interp.csr"):
                                from .ops import resize_from_m_to_t_csr
                                partial_attention_mask = resize_from_m_to_t_csr(
                                    partial_attention_mask, 0, k=self.pconfig.k, target_width=T_SRC, is_causal=False, benchmarking=True
                                )
                        elif SPARSITY_TYPE == 'coo':
                            with timer("interp.coo"):
                                partial_attention_mask = partial_attention_mask.reshape(N*H, T, T_M).to_sparse_coo()
                                partial_attention_mask = resize_width(partial_attention_mask, T/T_M)
                        else:
                            raise Exception()
                    else:
                        # TODO support causal sparse masking
                        SPARSITY_TYPE = 'flat_csr'
                        
                        if SPARSITY_TYPE == 'flat_csr':
                            from .ops import resize_from_m_to_t_csr
                            partial_attention_mask = resize_from_m_to_t_csr(
                                partial_attention_mask, 0, k=self.pconfig.k, target_width=T_SRC
                            )
                        elif SPARSITY_TYPE == 'coo':
                            partial_attention_mask = resize_from_m_to_t(partial_attention_mask, FP_MIN if not self.benchmarking else 0).view(N*H, T, T).to_sparse_coo()
                        else:
                            raise Exception()
                
                raise_if_nan(partial_attention_mask)
                    # TODO check what's wrong  #+ TODO any way to viz i.c.o self.benchmark?
                if not self.benchmarking:
                    partial_attention_mask_after_interp2 = partial_attention_mask.clone().permute(0,2,1,3).reshape(N, T, H*T)
                    get_bench().register_temp_buffer('partial_attention_mask_after_interp', partial_attention_mask_after_interp2)
            
            # return DUMMY_OUTPUT #1686
            
            if partial_attention_mask.is_sparse_csr:
                from .ops import flat_csr_to_dense
                get_bench().register_temp_buffer('partial_attention_mask', None, lazy=lambda: flat_csr_to_dense(partial_attention_mask, T_SRC, H))
            else:
                get_bench().register_temp_buffer('partial_attention_mask', partial_attention_mask)
            get_bench().register_temp_buffer('q_for_score', q_for_score)
            get_bench().register_temp_buffer('k_for_score', k_for_score)
            
            with timer("attention"):
                if not self.benchmarking:
                    # NOTE: checking avearge k is expected. uncomment following print, and then run visualize_glue
                    # avg_k_per_batch = (((partial_attention_mask > -1).view(N, -1).long().sum(-1) / (attention_mask > -1).long().view(N, -1).sum(-1)).mean() / H).item()
                    # #print(metric.update(avg_k_per_batch, name='avgk'))
                    
                    # TODO masked_fill rather than masked_fill_
                    attention_scores_dense = torch.matmul(q_for_score, k_for_score.transpose(-1, -2))
                    
                    # return DUMMY_OUTPUT #1774
                    
                    if attention_scores_truth is not None:
                        if not self.pconfig.causal:
                            attention_scores_dense = attention_scores_dense / math.sqrt(self.attention_head_size)
                            loss += kl_div_attention(
                                F.log_softmax(attention_scores_dense.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                                F.softmax(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                                attention_mask,
                            ) * 0.1
                            loss += F.mse_loss(
                                softmax_bf16(attention_scores_dense.masked_fill(attention_mask < -1, FP_MIN), dim=-1), 
                                softmax_bf16(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                            )
                        else:
                            attention_scores_dense = attention_scores_dense
                            with torch.autocast('cuda', torch.float32):
                                _t_causal_mask = causal_attention_mask < -1
                                # return DUMMY_OUTPUT #1778
                                _input = F.log_softmax(attention_scores_dense.masked_fill_(_t_causal_mask, FP_MIN).to(torch.float32), dim=-1, dtype=torch.float32).view(-1, attention_scores_dense.shape[-1])
                                # return DUMMY_OUTPUT #1970
                                _target = F.softmax(attention_scores_truth.masked_fill_(_t_causal_mask, FP_MIN).to(torch.float32), dim=-1, dtype=torch.float32).view(-1, attention_scores_dense.shape[-1])
                                # return DUMMY_OUTPUT #2162
                                loss += F.kl_div(
                                    _input,
                                    _target,
                                    reduction='batchmean',
                                ) * 0.1
                                del _input
                                # return DUMMY_OUTPUT #2738
                                loss += F.mse_loss(
                                    F.softmax(attention_scores_dense.to(torch.float32), dim=-1, dtype=torch.float32).view(-1, attention_scores_dense.shape[-1]), 
                                    _target,
                                )
                                del _target
                            # return DUMMY_OUTPUT #2738
                    get_bench().register_temp_buffer('attention_scores_dense', attention_scores_dense)
                    raise_if_nan(loss)
                    
                    # NOTE `attention_probs_dense` is for visualization, therefore it will not computed on benchmarking mode
                    if not self.pconfig.causal:
                        attention_scores_dense_masked = attention_scores_dense + attention_mask
                    else:
                        attention_scores_dense_masked = attention_scores_dense + causal_attention_mask
                    attention_probs_dense = softmax_bf16(attention_scores_dense_masked, dim=-1)
                    
                    #print('attention_probs_dense', attention_probs_dense)
                    #print('attention_probs_dense.shape', attention_probs_dense.shape)
                    attention_probs_dense_for_viz = attention_probs_dense.clone().permute(0,2,1,3).reshape(N, T, H*T)
                    get_bench().register_temp_buffer('attention_probs_dense', attention_probs_dense_for_viz) # 
                    # NOTE you should not add attention_mask and attention_score, because partial_attention_mask already has it.
                    raise_if_nan(partial_attention_mask)
                    partial_attention_scores = attention_scores_dense + partial_attention_mask # NOTE JIN masking pad should be included in partial_attention_mask
                    raise_if_nan(partial_attention_scores)
                    partial_attention_probs = softmax_bf16(partial_attention_scores, -1)
                    partial_attention_probs = partial_attention_probs * (partial_attention_mask > -1)
                    #print('partial_attention_scores', partial_attention_scores)
                    #print('partial_attention_scores.shape', partial_attention_scores.shape)
                    #print('partial_attention_probs', partial_attention_probs)
                    #print('partial_attention_probs.shape', partial_attention_probs.shape)
                    partial_attention_scores_for_viz = partial_attention_scores.clone().permute(0,2,1,3).reshape(N, T, H*T)
                    get_bench().register_temp_buffer('partial_attention_scores', partial_attention_scores_for_viz)
                    attention_matrix = partial_attention_probs.clone().permute(0,2,1,3).reshape(N, T, H*T)
                    get_bench().register_temp_buffer('attention_matrix', attention_matrix)
                    raise_if_nan(partial_attention_probs)
                    
                    # perform scaling, however this pervent to use spase attention kernel
                    estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                    if self.pconfig.partial_attention_scaler:
                        partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scales[..., 0:1])
                    
                    raise_if_nan(partial_attention_probs)
                    raise_if_nan(v)
                    partial_context_layer = torch.matmul(partial_attention_probs, v)
                    partial_context_layer_1 = partial_context_layer.clone()
                    get_bench().register_temp_buffer('partial_context_layer_1', partial_context_layer_1)
                else:
                    # TODO implement optimized causal kernel
                    # if self.pconfig.causal: raise Exception()
                    
                    attention_probs_dense = partial_attention_probs = attention_scores_dense = None
                    partial_context_layer = q_for_score
                    
                    # TODO HJ Apply probs scaler!
                    
                    # NOTE: #print avg k per batch
                    # avg_k_per_batch = (((partial_attention_mask.to_dense() > 0).view(N, -1).long().sum(-1) / (attention_mask > -1).long().view(N, -1).sum(-1)).mean() / H).item()
                    # print(metric.update(avg_k_per_batch, name='avgk'), flush=True)
                    
                    # using Numba
                    N, H, T, HEAD_H = q_for_score.shape
                    # #print((partial_attention_mask > -1).sum(), (partial_attention_mask > -1).sum() / partial_attention_mask.numel())
                    with mem("attention"):
                        if partial_attention_mask.is_sparse_csr:
                            from .ops import (
                                flat_csr_masked_bmm,
                                flat_csr_softmax,
                                flat_csr_elmul,
                                flat_csr_sdbmm,
                            )
                            with timer('attention.sparse.maksed_bmm'):
                                partial_attention_scores = flat_csr_masked_bmm(
                                    q_for_score, k_for_score, partial_attention_mask
                                )
                            with timer('attention.sparse.softmax'):
                                partial_attention_probs = flat_csr_softmax(
                                    partial_attention_scores, H, T_SRC
                                )
                            with timer('attention.sparse.scaler'):
                                estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                            with timer('attention.sparse.elmul'):
                                if self.pconfig.partial_attention_scaler:
                                    row_scaler = torch.sigmoid(estimated_scales[..., 0]).view(N, H, T_DST, 1).expand(N, H, T_DST, T_SRC)
                                    partial_attention_probs = flat_csr_elmul(partial_attention_probs, row_scaler)
                            with timer('attention.sparse.sdbmm'):
                                partial_context_layer = flat_csr_sdbmm(partial_attention_probs, v, T_M)
                        else:
                            with timer("attention.coo"), mem("attention.coo"):
                                if not partial_attention_mask.is_sparse:
                                    sparse_attention_mask = partial_attention_mask.float().view(N*H, T, T).to_sparse_coo()#.coalesce() 
                                else:
                                    sparse_attention_mask = partial_attention_mask#.coalesce() 
                            with timer("attention.sparse"), mem("attention.sparse"):
                                assert sparse_attention_mask._nnz() > 0, sparse_attention_mask
                                partial_attention_scores = sparse_attn(
                                    q_for_score.reshape(N*H, T, HEAD_H).contiguous(), 
                                    k_for_score.reshape(N*H, T, HEAD_H).contiguous(), 
                                    sparse_attention_mask
                                ).coalesce()
                                if not self.pconfig.causal:
                                    with timer("attention.sparse.score"):
                                        partial_attention_scores = partial_attention_scores / math.sqrt(self.attention_head_size)
                                get_bench().register_temp_buffer('partial_attention_scores', partial_attention_scores)
                                del sparse_attention_mask
                                # partial_attention_scores = partial_attention_scores.to_dense()[0].to_sparse_csr().to_sparse_bsr((1,1)).clone()
                            with timer("attention.sparse_softmax"), mem("attention.sparse_softmax"):
                                partial_attention_probs = torch.sparse.softmax(
                                    partial_attention_scores, dim=2
                                )
                            with timer('attention.sparse_scale'):
                                # partial_attention_probs = partial_attention_probs.to_dense()
                                estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                                if self.pconfig.partial_attention_scaler:
                                    partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scales[..., 0].view(N*H, T, 1))
                            with timer("attention.bmm"), mem("attention.bmm"):
                                partial_context_layer = torch.bmm(partial_attention_probs, v.reshape(N*H, T, HEAD_H))
                                partial_context_layer = partial_context_layer.view(N, H, T, HEAD_H)
                        
                # return DUMMY_OUTPUT #2782
                
                with timer("attention.avg_pool"):
                    if not self.pconfig.causal:
                        average_context_layer = (
                            v *\
                            (dst_attention_mask > -1).to(v.dtype) *\
                            resize_from_m_to_t(estimated_attention_probs.mean(-2, keepdim=True), 0, T).transpose(-1, -2)
                        ).sum(-2, keepdim=True).to(v.dtype)
                    else:
                        # TODO imporve this when causal
                        avg_v = v * (dst_attention_mask > -1)
                        average_context_layer = avg_v.cumsum(-2) / torch.arange(1, avg_v.shape[-2]+1, device=avg_v.device).view(1, 1, -1, 1)
                        average_context_layer = average_context_layer.to(v.dtype)
                        if average_context_layer.shape[-2] > q.shape[-2]:
                            average_context_layer = average_context_layer[...,-q.shape[-2]:,:]
                        # return DUMMY_OUTPUT #2978
                    average_scale = torch.sigmoid(estimated_scales[..., 1:2])
                    partial_context_layer = partial_context_layer * average_scale + (1-average_scale) * average_context_layer
                    get_bench().register_temp_buffer('estimated_scales', estimated_scales)
                    get_bench().register_temp_buffer('average_scale', average_scale)
                    if not self.pconfig.causal:
                        get_bench().register_temp_buffer('estimated_attention_probs_t', None, lazy=lambda: resize_from_m_to_t(estimated_attention_probs.mean(-2, keepdim=True), 0, T).transpose(-1, -2))
                    get_bench().register_temp_buffer('average_context_layer', average_context_layer)
                    partial_context_layer_2 = partial_context_layer.clone()
                    get_bench().register_temp_buffer('partial_context_layer_2', partial_context_layer_2)
            
            # return DUMMY_OUTPUT #2978
            
            if self.pconfig.random_lookup:
                raise Exception()
                # TODO please consider updated estimated attention probs
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

            with timer("context_permute"):
                partial_context_layer = partial_context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = partial_context_layer.size()[:-2] + (self.all_head_size,)
                partial_context_layer = partial_context_layer.view(new_context_layer_shape)
                if self.pconfig.out_add_performer_context:
                    performer_context_layer = performer_context_layer.permute(0, 2, 1, 3).contiguous()
                    performer_context_layer = performer_context_layer.view(new_context_layer_shape)
            
            # return DUMMY_OUTPUT #2978
            
            get_bench().register_temp_buffer('partial_context_layer_sparse', partial_context_layer)
            
            with timer("out"):
                if not self.pconfig.random_lookup:
                    normalized_partial_context_layer = self.norm_partial(partial_context_layer)
                    get_bench().register_temp_buffer('normalized_partial_context_layer', normalized_partial_context_layer)
                    
                    partial_context_layer = \
                        normalized_partial_context_layer +\
                        partial_context_layer
                    if self.pconfig.out_add_performer_context:
                        raise Exception('performer context hidden size is modified')
                        partial_context_layer = partial_context_layer +\
                            self.norm_performer(performer_context_layer)
                else:
                    raise Exception()
                    partial_context_layer = \
                        self.norm_partial(partial_context_layer) +\
                        self.norm_random(random_context_layer) +\
                        partial_context_layer
                    if self.pconfig.out_add_performer_context:
                        raise Exception('performer context hidden size is modified')
                        partial_context_layer = partial_context_layer +\
                            self.norm_performer(performer_context_layer)
                
                if self.pconfig.out_norm:
                    partial_context_layer = self.norm(partial_context_layer)
            
            # return DUMMY_OUTPUT #2978
            
            if not self.benchmarking:
                raise_if_nan(context_layer_truth)
                raise_if_nan(partial_context_layer)
                if context_layer_truth is not None:
                    loss += F.mse_loss(
                        context_layer_truth, 
                        partial_context_layer
                    )
                raise_if_nan(loss)

            raise_if_nan(loss)
            raise_if_nan(partial_context_layer)
            raise_if_nan(partial_attention_probs)
            raise_if_nan(attention_probs_dense)
            raise_if_nan(k_for_score)
            
            #print('partial_context_layer', partial_context_layer)
            #print('partial_context_layer.shape', partial_context_layer.shape)
            
            estimated_attention_probs_for_output = estimated_attention_probs if self.benchmarking else estimated_attention_probs_resized
            
            #print('estimated_attention_probs_for_output', estimated_attention_probs_for_output)
            #print('estimated_attention_probs_for_output.shape', estimated_attention_probs_for_output.shape)
            if not self.benchmarking:
                estimated_attention_probs_for_output_for_viz = estimated_attention_probs_for_output.clone().permute(0,2,1,3).reshape(N, T, H*T)
            else:
                estimated_attention_probs_for_output_for_viz = estimated_attention_probs_for_output.clone().permute(0,2,1,3).reshape(N, T, H*T_M)

            get_bench().register_temp_buffer('estimated_attention_probs_for_output', estimated_attention_probs_for_output_for_viz)
            get_bench().register_temp_buffer('partial_context_layer', partial_context_layer)
            
            assert partial_context_layer.shape[-2] == q.shape[-2]
            
            return PerlinAttentionOutput(
                loss=loss,
                context_layer=partial_context_layer,
                partial_attention_probs=partial_attention_probs,
                partial_attention_mask=partial_attention_mask,
                estimated_attention_probs_m = estimated_attention_probs,
                estimated_attention_probs=estimated_attention_probs_for_output,
                dense_attention_probs=attention_probs_dense,
                key_for_score=k_for_score,
                state=last_state,
            )
