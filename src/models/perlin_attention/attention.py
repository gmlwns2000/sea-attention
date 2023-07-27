import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .masked_mm import sparse_attn

import torch
import torch.nn.functional as F
from performer_pytorch import FastAttention
from torch import nn, optim
from xformers.components.attention.core import (
    SparseCS,
    scaled_dot_product_attention
)

from ...utils import get_bench, Metric
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
from ...utils import raise_if_nan
# NOTE HJ comment below to debug NaN
raise_if_nan = lambda x: x

timer = lambda name: get_bench().region(name)
mem = lambda name: get_bench().mem_region(name)
metric = Metric()

# NOTE HJ for temperaty development
T_MASK = None

def interpolate(x: torch.Tensor, size, interp_mode: str = None):
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

@dataclass
class PerlinAttentionOutput:
    loss: torch.Tensor
    context_layer: torch.Tensor
    partial_attention_probs: torch.Tensor
    estimated_attention_probs: torch.Tensor
    dense_attention_probs: torch.Tensor
    key_for_score: torch.Tensor

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
    def __init__(self, ch, padding=1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=padding, padding_mode='reflect'),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=padding, padding_mode='reflect'),
            # nn.BatchNorm2d(48),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_out = self.net(x)
        x = self.relu(x_out + x)
        return x

class UpsampleFP32(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        x_type = x.dtype
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if x_type != x.dtype:
            x = x.to(x_type)
        return x

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
        self.performer_nb_features = int(
            self.attention_head_size * math.log(self.attention_head_size) / self.pconfig.performer_nb_factor
        )
        self.performer = FastAttention(
            dim_heads = self.attention_head_size,
            nb_features = self.performer_nb_features,
            causal=False, # NOTE HJ if we handle causal attention, this should be changed.
        )
        self.performer_proj_updater = ProjectionUpdater(
            self.performer, 
            1000,
        )
        performer_value_hidden_size = self.attention_head_size*3
        self.attention_predictor_enc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(performer_value_hidden_size, self.attention_head_size*2),
            nn.LayerNorm(self.attention_head_size*2),
            nn.GELU(),
        )
        self.attention_predictor_dec_row = nn.Sequential(
            nn.Linear(self.attention_head_size*2, self.pconfig.attention_predictor_length),
        )
        self.attention_predictor_cnn = KeepRes(
            # NOTE if we use pixelshuffle outch should be 48
            nn.Conv2d(12, 48, 3, padding=0, stride=2),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.Conv2d(48, 48, 3, padding=1),
            # nn.BatchNorm2d(48),
            # nn.ReLU(),
            # nn.Conv2d(48, 48, 3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(48, 48, 3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(48, 48, 3, padding=1),
            # nn.ReLU(),
            ResBlock(48),
            ResBlock(48),
            # ResBlock(48),
            # nn.PixelShuffle(2),
            # nn.UpsamplingNearest2d(scale_factor=2),
            # UpsampleFP32(2),
            nn.ConvTranspose2d(48, 12, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=0),
        )
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
        # NOTE out linear is removed, following section is just for in case we revert this change...
        # self.out = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(self.all_head_size*2, config.hidden_size),
        #     nn.LayerNorm(config.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        # )
        # self.out_random_lookup = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(self.all_head_size*3, config.hidden_size),
        #     nn.LayerNorm(config.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        # )
        
        self.norm_performer = nn.LayerNorm(config.hidden_size)
        self.norm_partial = nn.LayerNorm(config.hidden_size)
        self.norm_random = nn.LayerNorm(config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        self.register_buffer('_v_eye', None, persistent=False)
        # self._v_eye = torch.eye(
        #     self.pconfig.v_eye_length, dtype=torch.float32
        # ).view(1, 1, self.pconfig.v_eye_length, self.pconfig.v_eye_length)
        
        self.v_eye_learned = nn.Parameter(
            data=torch.rand((1, 1, self.attention_head_size, self.attention_head_size)),
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
    ):
        if q.dtype in [torch.float16, torch.bfloat16]:
            # NOTE HJ even if we are in bfloat16, we have to use fp16 minimum because of F.interpolate
            FP_MIN = torch.finfo(torch.float16).min / 2
        elif q.dtype in [torch.float32]:
            FP_MIN = torch.finfo(torch.float32).min / 2
        else:
            raise Exception('unknown type')
        
        raise_if_nan(q)
        raise_if_nan(k)
        raise_if_nan(v)
        
        zero_one_attention_mask = (attention_mask > -1).float()
        zero_one_attention_mask_cumsum = zero_one_attention_mask.cumsum(-1)
        zero_one_attention_mask_sum = zero_one_attention_mask.sum(-1)
        
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
            N, H, T, HID = q.shape
            with timer("vmask"):
                # v_for_atten_identity = interpolate(
                #     x=self._v_eye,
                #     size=v_for_atten.shape[-2:],
                #     interp_mode='nearest'
                # ).expand(v_for_atten.shape).contiguous()
                with timer("vmaks.eye"):
                    # E_N = min(T, HID)
                    E_N = HID
                    
                    if self._v_eye is None or self._v_eye.shape[-1] != E_N:
                        v_for_atten_identity = torch.eye(
                            n=E_N,
                            dtype=v.dtype,
                            device=v.device,
                        ).view(1, 1, E_N, E_N)
                        self._v_eye = v_for_atten_identity
                    else:
                        v_for_atten_identity = self._v_eye
                    
                    # v_for_atten_identity = self.v_eye_learned
                    
                    v_for_atten_identity1 = interpolate( # NOTE just for visualization
                        x=self._v_eye,
                        size=v_for_atten.shape[-2:],
                        interp_mode='nearest'
                    ).expand(v_for_atten.shape).contiguous()
                    get_bench().register_temp_buffer('v_for_atten_identity_interpolate', v_for_atten_identity1)
                    
                    v_for_atten_identity = v_for_atten_identity.expand(v_for_atten.shape[:2] + (E_N, E_N))
                    get_bench().register_temp_buffer('v_for_atten_identity_bef_grid', v_for_atten_identity)
                
                with timer("vmask.grid"):
                    token_index_y = ((zero_one_attention_mask_cumsum - 1.0) / ((zero_one_attention_mask_sum - 1.0).view(N, 1, 1, 1) + 1e-8) * 2 - 1)\
                        .view(N, T, 1, 1)\
                        .expand(N, T, HID, 1)
                    # if self._v_grid_x is None or self._v_grid_x.shape[-2] != HID:
                    #     token_index_x = (torch.arange(HID, device=q.device, dtype=q.dtype) / (HID - 1) * 2 - 1).view(1, 1, HID, 1)
                    #     self._v_grid_x = token_index_x
                    # else:
                    #     token_index_x = self._v_grid_x
                    token_index_x = (torch.arange(HID, device=q.device, dtype=q.dtype) / (HID - 1) * 2 - 1).view(1, 1, HID, 1)
                    token_index_x = token_index_x.expand(N, T, HID, 1)
                    token_index = torch.cat([token_index_x, token_index_y], dim=-1)
                
                with timer("vmask.sample"):
                    v_for_atten_identity = F.grid_sample(
                        input=v_for_atten_identity, 
                        grid=token_index, 
                        mode='bilinear',
                        align_corners=True,
                    )
                    get_bench().register_temp_buffer('v_for_atten_identity_aft_grid', v_for_atten_identity)
                    
                with timer("vmask.cat_fill"):
                    v_for_atten = torch.cat([
                        v_for_atten_identity, 
                        v_for_atten
                    ], dim=-1)

                    # NOTE JIN duplicated calculation
                    v_for_atten.masked_fill_(attention_mask.transpose(-1, -2) < -1, 0)
                    v.masked_fill_(attention_mask.transpose(-1, -2) < -1, 0)
            
            with timer("performer"):
                if not self.benchmarking:
                    q_type = q_for_atten.dtype
                    with torch.autocast('cuda', torch.float32):
                        performer_context_layer = self.performer(
                            q_for_atten, 
                            k_for_atten, 
                            v_for_atten
                        )
                    if q_type != performer_context_layer.dtype:
                        performer_context_layer = performer_context_layer.to(q_type)
                else:
                    # TODO: fix numerical stability...
                    performer_context_layer = self.performer(
                        q_for_atten, 
                        k_for_atten, 
                        v_for_atten
                    )
                get_bench().register_temp_buffer('performer_context_layer', performer_context_layer)
                get_bench().register_temp_buffer('performer_context_layer>0', performer_context_layer>0)
                
            with timer("performer_value"):
                # NOTE HJ Cut gradient from loss_sp, because loss_sp has negative effect to loss_model when approximation is sucks.
                performer_value = torch.cat([
                    performer_context_layer, 
                    v
                ], dim=-1)#.detach()
                raise_if_nan(performer_value)
            
            # estimate attention scores
            with timer("predictor"):
                if self.pconfig.attention_predictor_method == 'mlp':
                    raise_if_nan(performer_value)
                    t_attention_predictor = self.attention_predictor_enc(performer_value)
                    raise_if_nan(t_attention_predictor)
                    estimated_attention_score = self.attention_predictor_dec_row(t_attention_predictor) # type: torch.Tensor
                    raise_if_nan(estimated_attention_score)
                    estimated_attention_score = self.attention_predictor_cnn(estimated_attention_score)
                    raise_if_nan(estimated_attention_score)
                elif self.pconfig.attention_predictor_method == 'comp':
                    warnings.warn('attention prediction method is compressed one.')
                    t_attention_predictor = self.attention_predictor_comp_enc(performer_value)
                    estimated_attention_score = self.attention_predictor_comp_dec_row(t_attention_predictor)
                    estimated_attention_score = estimated_attention_score\
                        .view(N, H, T, self.pconfig.attention_predictor_comp_patch_count, self.pconfig.attention_predictor_comp_book_size)
                    _, _, _, CODE_SEQ_LEN, BOOK_LEN = estimated_attention_score.shape
                    estimated_attention_score = torch.softmax(estimated_attention_score, dim = -1)
                    estimated_attention_score = torch.matmul(
                        estimated_attention_score.view(-1, BOOK_LEN), 
                        self.attention_predictor_comp_codebook
                    )
                    estimated_attention_score = estimated_attention_score.view(N, H, T, -1)
                else:
                    raise Exception()
            
            # interpolate and convert to probability
            with timer("mask_softmax"):
                T_M = estimated_attention_score.shape[-1]
                # TODO: this should be grid sample to T, T
                # resized_attention_mask = interpolate(
                #     x=attention_mask, 
                #     size=(1, T_M), 
                #     interp_mode='nearest',
                # )
                
                # token_index_x = (resized_attention_mask > -1).float().view(N, 1, T_M)
                # token_index_x = token_index_x.cumsum(-1) - 1.0
                # token_index_x = (token_index_x / T * 2 - 1).expand(N, T, T_M)
                # token_index_y = (
                #     torch.arange(T, dtype=token_index_x.dtype, device=token_index_x.device)\
                #         .view(1, T, 1) / T * 2 - 1)\
                #         .expand(N, T, T_M)
                # token_index = torch.cat([token_index_x.unsqueeze(-1), token_index_y.unsqueeze(-1)], dim=-1)
                # estimated_attention_score = F.grid_sample(
                #     input=estimated_attention_score, 
                #     grid=token_index,
                #     mode='nearest'
                # )
                
                # resized_attention_mask_binary = resized_attention_mask < -1
                # # resized_attention_mask = (resized_attention_mask < -1) * FP_MIN
                # if not self.benchmarking:
                #     estimated_attention_score_unmasked = estimated_attention_score
                #     estimated_attention_score = estimated_attention_score.masked_fill(
                #         mask=resized_attention_mask_binary,
                #         value=FP_MIN
                #     )
                # else:
                #     estimated_attention_score = estimated_attention_score.masked_fill_(
                #         mask=resized_attention_mask_binary,
                #         value=FP_MIN
                #     )
                
                # token_index_x = (attention_mask > -1).float().view(N, 1, T)
                # token_index_x = token_index_x.cumsum(-1) - 1.0
                # token_index_x = (token_index_x / ((attention_mask > -1).float().sum(-1).view(N, 1, 1) + 1e-6) * 2 - 1).expand(N, T, T)
                # token_index_y = (
                #     torch.arange(T, dtype=token_index_x.dtype, device=token_index_x.device)\
                #         .view(1, T, 1) / T * 2 - 1)\
                #         .expand(N, T, T)
                # token_index = torch.cat([token_index_x.unsqueeze(-1), token_index_y.unsqueeze(-1)], dim=-1)
                # estimated_attention_score = F.grid_sample(
                #     input=estimated_attention_score, 
                #     grid=token_index,
                #     mode='nearest'
                # )
                # estimated_attention_score_unmasked = estimated_attention_score
                # estimated_attention_score = estimated_attention_score.masked_fill_(
                #     mask=attention_mask < -1,
                #     value=FP_MIN
                # )
                
                estimated_attention_probs = torch.softmax(estimated_attention_score, -1)
            
            get_bench().register_temp_buffer('estimated_attention_probs', estimated_attention_probs)
            
            # in layerwise, train perlin attention predictor
            def resize_from_m_to_t(x, masked_fill_value, target_width=None):
                N, H, T1, T_M = x.shape
                if target_width is not None:
                    T2 = target_width
                else:
                    T2 = T1
                with timer("resize"):
                    with timer("resize.grid"):
                        token_index_x = zero_one_attention_mask.view(N, 1, T2)
                        if masked_fill_value is not None:
                            token_index_x = torch.roll(token_index_x, shifts=(1,), dims=(-1)).cumsum(-1) + ((1.0 - zero_one_attention_mask) * 2).view(N, 1, T2)
                            token_index_x = (token_index_x / ((zero_one_attention_mask.sum(-1) + 2).view(N, 1, 1) + 1e-8) * 2 - 1).expand(N, T1, T2)
                        else:
                            token_index_x = token_index_x.cumsum(-1)
                            token_index_x = (token_index_x / ((zero_one_attention_mask.sum(-1) - 1).view(N, 1, 1) + 1e-8) * 2 - 1).expand(N, T1, T2)
                        token_index_y = (
                            torch.arange(T1, dtype=token_index_x.dtype, device=token_index_x.device)\
                                .view(1, T1, 1) / T1 * 2 - 1)\
                                .expand(N, T1, T2) #type: torch.Tensor
                        token_index = torch.cat([
                            token_index_x.unsqueeze(-1), 
                            token_index_y.unsqueeze(-1)
                        ], dim=-1)
                    
                    with timer("resize.sample"):
                        return F.grid_sample(
                            input=F.pad(F.pad(x, pad=(0, 2), value=0), pad=(0, 1), value=masked_fill_value) if masked_fill_value is not None else x,
                            grid=token_index,
                            mode='nearest',
                            align_corners=True,
                            padding_mode='border'
                        )
            
            loss = 0
            if not self.benchmarking:
                # for loss calculation
                estimated_attention_probs_resized = resize_from_m_to_t(estimated_attention_probs, masked_fill_value=0)
                estimated_attention_score_resized = resize_from_m_to_t(estimated_attention_score, masked_fill_value=FP_MIN)
                
                with torch.autocast('cuda', torch.float32):
                    raise_if_nan(estimated_attention_score_resized)
                    raise_if_nan(attention_scores_truth)
                    raise_if_nan(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN))
                    raise_if_nan(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN))
                    loss_kl = kl_div_attention(
                        F.log_softmax(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                        F.softmax(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                        attention_mask,
                    ) * 0.1
                    raise_if_nan(loss_kl)
                    loss_mse = F.mse_loss(
                        torch.softmax(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN), dim=-1), 
                        torch.softmax(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1)
                    )
                    raise_if_nan(loss_mse)
                    loss += loss_kl + loss_mse
                    raise_if_nan(loss)
            
            with timer("mask"):
                estimated_attention_probs = estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1)
                
                T_M = estimated_attention_probs.shape[-1]
                token_length = (attention_mask > -1).long().sum(-1).view(N, -1)
                top_k = min(max(int(round(self.pconfig.k * (T_M / min(token_length).item()))), 1), T_M) # TODO topk is quiet strange..
                # top_k = min(max(int(round(self.pconfig.k * (T_M / T))), 1), T_M) # NOTE JIN fixed top_k..should be 7 per T not 7 per Ti..
                k_flatten = self.pconfig.k_flatten
                if k_flatten:
                    k_flatten_dim = self.pconfig.k_flatten_dim
                    assert k_flatten_dim in ['head', 'batch']
                perlin_col_select = self.pconfig.colsel
                if perlin_col_select and k_flatten:
                    col_select_method = self.pconfig.colsel_method
                    mask_in_probs = self.pconfig.colsel_mask_in_probs # "scores"
                    warnings.warn(f'perlin_col_select {perlin_col_select}')
                    selected_col_per_head = 1 # TODO add to args
                    warnings.warn(f"selected_col_per_head {selected_col_per_head}")
                    top_k = top_k-selected_col_per_head # NOTE not that important, this doesn't control final topk value
                    with timer("mask.select_col"):
                        if k_flatten_dim == "batch":
                            selected_col_cnt = self.num_attention_heads*selected_col_per_head # TODO int(*(T_M/T))
                            warnings.warn(f"selected col cnt {selected_col_cnt}")
                            warnings.warn(f"col_select_method {col_select_method}")
                            warnings.warn(f"mask_in_probs {mask_in_probs}")
                            col_sel_estimated_attention_probs1 = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1)).permute(0, 2, 1, 3).contiguous().view(N, T, H*T_M) # NOTE 'batch': memory order differs depending on select col T/F
                            col_sel_estimated_attention_probs = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1)).permute(0, 2, 1, 3).contiguous().view(N, T, H*T_M) # NOTE 'batch': memory order differs depending on select col T/F
                            get_bench().register_temp_buffer('col_sel_estimated_attention_probs_bef_select', col_sel_estimated_attention_probs1) # col_sel_estimated_attention_probs1
                            # breakpoint()
                            if col_select_method == "sum_values":
                                sum_per_col = col_sel_estimated_attention_probs.sum(dim=-2) # [N, H*T_M]
                            elif col_select_method == "sum_mask":
                                col_select_mask = col_sel_estimated_attention_probs >= (1/T_M) # [N, T, H*T_M]
                                sum_per_col = col_select_mask.sum(dim=-2) # [N, H*T_M]
                            # get_bench().register_temp_buffer('sum_per_col', sum_per_col)
                            # breakpoint()
                            _, largest_indx = torch.topk(
                                input=sum_per_col,
                                k=selected_col_cnt,
                                dim=-1
                            ) # [N, selected_col_cnt] 0~H*T_M-1 in each row
                            # breakpoint()
                            large_inx = largest_indx.view(N, 1, selected_col_cnt)\
                                .expand(N, T, selected_col_cnt) # [N, T, selected_col_cnt]
                            # breakpoint()
                            if mask_in_probs:
                                col_sel_estimated_attention_probs.scatter_(dim=-1, index=large_inx, value=0) # this also changes estimated_attention_probs
                            else:
                                col_sel_estimated_attention_score = estimated_attention_score.permute(0, 2, 1, 3).contiguous().view(N, T, H*T_M) # NOTE attn mask not applied
                                col_sel_estimated_attention_score.scatter_(dim=-1, index=large_inx, value=FP_MIN)
                                col_sel_estimated_attention_score = col_sel_estimated_attention_score.view(N, T, H, T_M).permute(0, 2, 1, 3)
                                col_sel_estimated_attention_probs = torch.softmax(col_sel_estimated_attention_score, -1)
                                col_sel_estimated_attention_probs = col_sel_estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1) # N H T T_M
                                col_sel_estimated_attention_probs = col_sel_estimated_attention_probs.permute(0, 2, 1, 3).contiguous()

                            get_bench().register_temp_buffer('col_sel_estimated_attention_probs', col_sel_estimated_attention_probs)
                            t = col_sel_estimated_attention_probs.view(N, T*H*T_M) # [N, T, H*T_M] -> [N, T*H*T_M]
                        elif k_flatten_dim == 'head':
                            # 1 per head # TODO change head code
                            selected_col_cnt = selected_col_per_head
                            sum_per_col = estimated_attention_probs[:,:,:,:].sum(dim=2) # [N, H, T, T_M] -> [N, H, T_M]
                            largest_indx = sum_per_col.argmax(dim=-1) # would be [N, H] 0~T_M-1 in each row
                            large_inx = largest_indx.view(N, H, 1, 1)\
                                .expand_as(estimated_attention_probs) # [N, H, T, T_M]
                            inx = torch.arange(
                                T_M,
                                dtype=large_inx.dtype, # TODO check dtype device
                                device=large_inx.device,
                                ).view(1,1,1,T_M)\
                                .expand_as(estimated_attention_probs) # [N, H, T, T_M]
                            col_select_mask = (inx == large_inx) # TODO check device
                            estimated_attention_probs[col_select_mask] = 0.0 # [N, H, T, T_M]
                            if k_flatten and k_flatten_dim == 'head':
                                t = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1)).view(N, H, T*T_M) # TODO check for head
                        else:
                            raise Exception(f'k_flatten_dim {k_flatten_dim}')
                    
                if not k_flatten:
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
                    warnings.warn(f"k_flatten_dim {k_flatten_dim}")
                    if not perlin_col_select:
                        with timer("mask.view"):
                            if k_flatten_dim == 'batch':
                                t = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1)).view(N, H*T*T_M) # TODO memory order different to the case of select col?
                            elif k_flatten_dim == 'head':
                                t = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1)).view(N, H, T*T_M) # TODO check for head
                            else:
                                raise Exception(f'k_flatten_dim {k_flatten_dim}')                        
                    with timer("mask.topk"):
                        _, indices = torch.topk(
                            input=t,
                            k=top_k*T*H if k_flatten_dim == 'batch' else top_k*T, 
                            dim=-1, 
                            sorted=True #sorted true is important
                        )
                        # breakpoint()
                    # with timer("mask.empty"):
                    #     partial_attention_mask = torch.empty(
                    #         t.shape, 
                    #         dtype=q_for_score.dtype, 
                    #         device=attention_mask.device,
                    #     )
                    # with timer("mask.fill"):
                    #     partial_attention_mask.fill_(FP_MIN)
                    # with timer("mask.scatter"):
                    #     partial_attention_mask.scatter_(dim=-1, index=indices, value=0)
                    with timer("mask.empty"):
                        partial_attention_mask = torch.empty(
                            t.shape, 
                            dtype=torch.long, 
                            device=attention_mask.device,
                        )
                        # breakpoint()
                    with timer("mask.fill"):
                        partial_attention_mask.fill_(t.shape[-1])
                        # breakpoint()
                    with timer("mask.scatter"):
                        partial_attention_mask.scatter_(
                            dim=-1,
                            index=indices,
                            src=torch.arange(
                                top_k*T*H if k_flatten_dim == 'batch' else top_k*T, 
                                dtype=torch.long,
                                device=attention_mask.device, 
                            )\
                                .view((1, -1) if k_flatten_dim == 'batch' else (1, 1, -1))\
                                .expand(indices.shape)
                        )
                        # breakpoint()
                        # print(partial_attention_mask[0].view(H, T, T_M)[0])
                    with timer("mask.masked_fill"):
                        # input()
                        if perlin_col_select:
                            per_item_top_k = token_length * (H if k_flatten_dim == 'batch' else 1) * (self.pconfig.k-torch.floor(selected_col_per_head*(token_length/T_M))) * torch.ceil(T_M / token_length) # TODO check value
                            # per_item_top_k = token_length * (H if k_flatten_dim == 'batch' else 1) * math.ceil(self.pconfig.k-math.floor(selected_col_per_head*(T/T_M))) * torch.ceil(T_M / token_length) # TODO check value
                            # per_item_top_k = token_length * (H if k_flatten_dim == 'batch' else 1) * int(self.pconfig.k-round(selected_col_per_head*(T/T_M))) * torch.ceil(T_M / token_length) # TODO check value
                        else:
                            per_item_top_k = token_length * (H if k_flatten_dim == 'batch' else 1) * self.pconfig.k * torch.ceil(T_M / token_length)
                            # per_item_top_k = token_length * (H if k_flatten_dim == 'batch' else 1) * self.pconfig.k * torch.floor(T / token_length) TODO test this value
                        # breakpoint()
                        # print((token_length * (top_k * H if k_flatten_dim == 'batch' else top_k)).view(-1), token_length.view(-1), top_k, self.pconfig.k, (T_M/token_length).view(-1), per_item_top_k.view(-1))
                        # t_dead_mask = partial_attention_mask >= (token_length * (top_k * H if k_flatten_dim == 'batch' else top_k)) #k is resized
                        if not self.benchmarking:
                            t_dead_mask = partial_attention_mask >= per_item_top_k
                            # breakpoint()
                            # partial_attention_mask.fill_(FP_MIN)
                            # partial_attention_mask.masked_fill_(t_alive_mask, value=0)
                            partial_attention_mask = t_dead_mask.to(q.dtype) * FP_MIN
                            # breakpoint()
                        else:
                            t_alive_mask = partial_attention_mask < per_item_top_k
                            partial_attention_mask = t_alive_mask.float()
                    if perlin_col_select:
                        if k_flatten_dim=='batch': # partial_attention_mask [N, T*H*T_M]
                            # breakpoint()
                            partial_attention_mask = partial_attention_mask.view(N, T, H*T_M) # NOTE memory order is carefully considered
                            # breakpoint()
                            partial_attention_mask.scatter_(dim=-1, index=large_inx, value=0.0) # [N, T, selected_col_cnt] ~> (N, T, H*T_M)
                            # breakpoint()
                            partial_attention_mask = partial_attention_mask.view(N, T, H, T_M).permute(0,2,1,3)
                            # breakpoint()
                        elif k_flatten_dim=="head":
                            partial_attention_mask = partial_attention_mask.view(N, H, T, T_M)
                            partial_attention_mask[col_select_mask] = 0.0
                        else:
                            raise Exception(f'k flatten {k_flatten} k_flatten_dim {k_flatten_dim}')
                    else:
                        partial_attention_mask = partial_attention_mask.view(N, H, T, T_M) # NOTE memory order considered
                        # breakpoint()
                    assert partial_attention_mask.shape ==(N, H, T, T_M)
                    
            
            get_bench().register_temp_buffer('partial_attention_mask_before_interp', partial_attention_mask)
            
            with timer("interp"):
                # NOTE: partial attention mask should be filled with 0 and -inf only.
                raise_if_nan(partial_attention_mask)
                # partial_attention_mask = interpolate(
                #     x=partial_attention_mask, 
                #     size=(T, T), 
                #     interp_mode='nearest'
                # )
                if not self.benchmarking:
                    with timer("interp.resize"):
                        # print(torch.unique(partial_attention_mask).shape)
                        # TODO Fix this function to return COO tensor
                        partial_attention_mask = resize_from_m_to_t(partial_attention_mask, FP_MIN if not self.benchmarking else 0)
                        # print(torch.unique(partial_attention_mask).shape)
                        # print(partial_attention_mask[0,0,0,-10:], attention_mask[0,0,0,-10:])
                        # input()
                else:
                    def resize_width(img: torch.Tensor, scale: float):
                        N, H, W = img.shape
                        img = img.coalesce()
                        idx = img.indices().float()
                        nnz = idx.shape[-1]
                        if scale < 1.0:
                            xs_scaled = idx[2] * scale
                            xs_rounded = torch.clamp(torch.round(xs_scaled), 0, round(W*scale)-1)
                            xs_okay = torch.abs(xs_rounded - xs_scaled) < (scale * 0.5)
                            idx[2] = xs_rounded * xs_okay
                            idx = idx * xs_okay.unsqueeze(0) + (~xs_okay.unsqueeze(0)) * -1
                            idx = torch.unique(idx.long(), dim=-1)
                            idx = idx[:, 1:]
                            
                            # print(nnz, idx.shape[-1])
                            return torch.sparse_coo_tensor(
                                indices=idx,
                                values=torch.ones((idx.shape[-1],), device=img.device, dtype=img.dtype),
                                size=(N, H, round(W*scale)),
                            )
                        elif scale > 1.0:
                            scale_ceil = math.ceil(scale)
                            idx = F.interpolate(idx.view(1, 1, 3, nnz), size=(3, nnz*scale_ceil), mode='nearest').view(3, nnz*scale_ceil)
                            idx[2] = idx[2] * scale_ceil + torch.arange(nnz*scale_ceil, device=img.device) % scale_ceil
                            
                            shrink_scale = scale / scale_ceil
                            xs_scaled = idx[2] * shrink_scale
                            xs_rounded = torch.round(xs_scaled)
                            xs_okay = torch.abs(xs_rounded - xs_scaled) < (shrink_scale * 0.5)
                            del xs_scaled
                            idx[2] = xs_rounded * xs_okay
                            idx[2] = torch.clamp(idx[2], 0, round(W*scale)-1)
                            idx = idx * xs_okay.unsqueeze(0) + (~xs_okay.unsqueeze(0)) * -1
                            del xs_okay
                            idx = torch.unique(idx.long(), dim=-1)
                            idx = idx[:, 1:]
                            # idx = torch.unique(idx.long(), dim=-1)
                            
                            # print(nnz, idx.shape[-1])
                            return torch.sparse_coo_tensor(
                                indices=idx,
                                values=torch.ones((idx.shape[-1],), device=img.device, dtype=img.dtype),
                                size=(N, H, round(W*scale)),
                            )
                        else:
                            return img

                    N, H, T, T_M = partial_attention_mask.shape
                    
                    # original
                    # partial_attention_mask_original = resize_from_m_to_t(partial_attention_mask, FP_MIN if not self.benchmarking else 0).view(N*H, T, T).to_sparse_coo()
                    
                    # optimized
                    partial_attention_mask = partial_attention_mask.view(N*H, T, T_M).to_sparse_coo()
                    partial_attention_mask = resize_width(partial_attention_mask, T/T_M)
                    
                    # torch.save({
                    #     'a':partial_attention_mask_original,
                    #     'b':partial_attention_mask
                    # }, "help.pth")
                
                # with timer("interp.fill"):
                #     partial_attention_mask.masked_fill_(
                #         mask=attention_mask < -1,
                #         value=FP_MIN,
                #     )
                raise_if_nan(partial_attention_mask)
            
            get_bench().register_temp_buffer('partial_attention_mask', partial_attention_mask)
            
            with timer("attention"):
                if not self.benchmarking:
                    # start of masked attention mechanism
                    
                    # NOTE: checking avearge k is expected. uncomment following print, and then run visualize_glue
                    # print(
                    #     (partial_attention_mask > -1).long().sum(-1).sum(-1)[:,0].view(-1),
                    #     (attention_mask > -1).long().sum(-1).view(-1),
                    #     (partial_attention_mask > -1).long().sum(-1).sum(-1)[:,0].view(-1) / (attention_mask > -1).long().sum(-1).view(-1),
                    #     k, T, T_M
                    # )
                    # NOTE: print avg k per batch
                    # avg_k_per_batch = (((partial_attention_mask > -1).view(N, -1).long().sum(-1) / (attention_mask > -1).long().view(N, -1).sum(-1)).mean() / H).item()
                    # print(metric.update(avg_k_per_batch, name='avgk'))
                    
                    attention_scores_dense = torch.matmul(q_for_score, k_for_score.transpose(-1, -2))
                    attention_scores_dense = attention_scores_dense / math.sqrt(self.attention_head_size)
                    loss += kl_div_attention(
                        F.log_softmax(attention_scores_dense.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                        F.softmax(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                        attention_mask,
                    ) * 0.1
                    loss += F.mse_loss(
                        torch.softmax(attention_scores_dense.masked_fill(attention_mask < -1, FP_MIN), dim=-1), 
                        torch.softmax(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                    )
                    
                    raise_if_nan(loss)
                    
                    # NOTE HJ `attention_probs_dense` is for visualization, therefore it will not computed on benchmarking mode
                    if attention_mask is not None:
                        attention_scores_dense_masked = attention_scores_dense + attention_mask
                    attention_probs_dense = torch.softmax(attention_scores_dense_masked, dim=-1)
                    
                    get_bench().register_temp_buffer('attention_probs_dense', attention_probs_dense)
                    # NOTE HJ you should not add attention_mask and attention_score, because partial_attention_mask already has it.
                    # print(
                    #     torch.unique((partial_attention_mask).view(-1)), 
                    #     torch.max(attention_scores_dense.view(-1)), 
                    #     torch.min(attention_scores_dense.view(-1)),
                    #     attention_scores_dense.dtype, partial_attention_mask.dtype
                    # )
                    raise_if_nan(partial_attention_mask)
                    partial_attention_scores = attention_scores_dense + partial_attention_mask
                    raise_if_nan(partial_attention_scores)
                    partial_attention_probs = torch.softmax(partial_attention_scores, -1)
                    partial_attention_probs = partial_attention_probs * (partial_attention_mask > -1)
                    get_bench().register_temp_buffer('attention_matrix', partial_attention_probs)
                    raise_if_nan(partial_attention_probs)
                    
                    # perform scaling, however this pervent to use spase attention kernel
                    estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                    if self.pconfig.partial_attention_scaler:
                        partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scales[..., 0:1])
                    
                    raise_if_nan(partial_attention_probs)
                    raise_if_nan(v)
                    partial_context_layer = torch.matmul(partial_attention_probs, v)
                    
                    average_context_layer = (
                        v *\
                        (attention_mask.transpose(-1, -2) > -1) *\
                        resize_from_m_to_t(estimated_attention_probs.mean(-2, keepdim=True), 0, T).transpose(-1, -2)
                    ).sum(-2, keepdim=True)
                    average_scale = torch.sigmoid(estimated_scales[..., 1:2])
                    partial_context_layer = partial_context_layer * average_scale + (1-average_scale) * average_context_layer
                    
                    # average_context_layer = (v * (attention_mask.transpose(-1, -2) > -1)).sum(-2, keepdim=True) /\
                    #     (attention_mask > -1).float().sum(-1, keepdim=True)
                    # average_scale = torch.sigmoid(estimated_scales[..., 1:2])
                    # partial_context_layer = partial_context_layer * average_scale + (1-average_scale) * average_context_layer
                else:
                    attention_probs_dense = partial_attention_probs = attention_scores_dense = None
                    partial_context_layer = q_for_score
                    
                    # TODO HJ Apply probs scaler!
                    
                    # NOTE: print avg k per batch
                    # avg_k_per_batch = (((partial_attention_mask.to_dense() > 0).view(N, -1).long().sum(-1) / (attention_mask > -1).long().view(N, -1).sum(-1)).mean() / H).item()
                    # print(metric.update(avg_k_per_batch, name='avgk'))
                    
                    # using xFormers
                    # with timer("attention.binary_mask"):
                    #     sparse_attention_mask = partial_attention_mask < -1
                    # N, H, T, HEAD_H = q_for_score.shape
                    # with timer("attention.sparsify"):
                    #     global T_MASK
                    #     if T_MASK is None:
                    #         T_MASK = SparseCS(
                    #             sparse_attention_mask.view(N*H, T, T)[:1, :, :],
                    #             device=q_for_score.device
                    #         )
                    #     sparse_mask = T_MASK
                    # with timer("attention.attention"):
                    #     partial_context_layer = scaled_dot_product_attention(
                    #         q=q_for_score.reshape(N*H, T, HEAD_H),
                    #         k=k_for_score.reshape(N*H, T, HEAD_H),
                    #         v=v.reshape(N*H, T, HEAD_H),
                    #         att_mask=sparse_mask
                    #     )
                    # partial_context_layer = partial_context_layer.view(N, H, T, HEAD_H)
                    
                    # using Numba
                    N, H, T, HEAD_H = q_for_score.shape
                    # print((partial_attention_mask > -1).sum(), (partial_attention_mask > -1).sum() / partial_attention_mask.numel())
                    with mem("attention"):
                        with timer("attention.coo"), mem("attention.coo"):
                            if not partial_attention_mask.is_sparse:
                                sparse_attention_mask = partial_attention_mask.float().view(N*H, T, T).to_sparse_coo()
                            else:
                                sparse_attention_mask = partial_attention_mask
                            del partial_attention_mask
                        with timer("attention.sparse"), mem("attention.sparse"):
                            # print(torch.min(q_for_score))
                            # print(torch.min(k_for_score))
                            # print(torch.min(sparse_attention_mask))
                            partial_attention_scores = sparse_attn(
                                q_for_score.reshape(N*H, T, HEAD_H).contiguous(), 
                                k_for_score.reshape(N*H, T, HEAD_H).contiguous(), 
                                sparse_attention_mask
                            ) / math.sqrt(self.attention_head_size)
                            # print(partial_attention_scores)
                            get_bench().register_temp_buffer('partial_attention_scores', partial_attention_scores)
                            del sparse_attention_mask
                        with timer("attention.sparse_softmax"), mem("attention.sparse_softmax"):
                            partial_attention_probs = torch.sparse.softmax(
                                partial_attention_scores, dim=2
                            )
                            get_bench().register_temp_buffer('attention_matrix', partial_attention_probs)
                            del partial_attention_scores
                            estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                            if self.pconfig.partial_attention_scaler:
                                partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scales[..., 0].view(N*H, T, 1))
                            
                        with timer("attention.bmm"), mem("attention.bmm"):
                            partial_context_layer = torch.bmm(partial_attention_probs, v.reshape(N*H, T, HEAD_H))
                            partial_context_layer = partial_context_layer.view(N, H, T, HEAD_H)
                        
                        with timer("attention.avg_pool"):
                            average_context_layer = (
                                v *\
                                (attention_mask.transpose(-1, -2) > -1) *\
                                resize_from_m_to_t(estimated_attention_probs.mean(-2, keepdim=True), 0, T).transpose(-1, -2)
                            ).sum(-2, keepdim=True)
                            average_scale = torch.sigmoid(estimated_scales[..., 1:2])
                            partial_context_layer = partial_context_layer * average_scale + (1-average_scale) * average_context_layer
            
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
            
            with timer("out"):
                if not self.pconfig.random_lookup:
                    partial_context_layer = \
                        self.norm_partial(partial_context_layer) +\
                        partial_context_layer
                    if self.pconfig.out_add_performer_context:
                        raise Exception('performer context hidden size is modified')
                        partial_context_layer = partial_context_layer +\
                            self.norm_performer(performer_context_layer)
                else:
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
            
            if not self.benchmarking:
                raise_if_nan(context_layer_truth)
                raise_if_nan(partial_context_layer)
                loss += F.mse_loss(
                    context_layer_truth, 
                    partial_context_layer
                )
                raise_if_nan(loss)

            raise_if_nan(loss)
            raise_if_nan(partial_context_layer)
            raise_if_nan(partial_attention_probs)
            # raise_if_nan(estimated_attention_probs_resized)
            raise_if_nan(attention_probs_dense)
            raise_if_nan(k_for_score)
            
            estimated_attention_probs_for_output = estimated_attention_probs if self.benchmarking else estimated_attention_probs_resized
            get_bench().register_temp_buffer('estimated_attention_probs_for_output', estimated_attention_probs_for_output)
            
            return PerlinAttentionOutput(
                loss=loss,
                context_layer=partial_context_layer,
                partial_attention_probs=partial_attention_probs,
                estimated_attention_probs=estimated_attention_probs_for_output,
                dense_attention_probs=attention_probs_dense,
                key_for_score=k_for_score,
            )
