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
from .modules import (
    ResBlock,
    Residual,
    KeepRes,
    CausalConv2d,
    UpsampleFP32,
    interpolate
)
from math import ceil, floor
# NOTE HJ comment below to debug NaN
raise_if_nan = lambda x: x

timer = lambda name: get_bench().region(name)
mem = lambda name: get_bench().mem_region(name)
metric = Metric()

# NOTE HJ for temperaty development
T_MASK = None

def grid_sample_bf16(input, grid, mode='nearest', align_corners=False, padding_mode='zeros'):
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
    if y.dtype != input_dtype:
        y = y.to(input_dtype)
    return y

def softmax_bf16(input, dim=-1):
    input_dtype = input.dtype
    op_dtype = torch.float32 if torch.get_autocast_gpu_dtype() in [torch.bfloat16, torch.float16] else input_dtype
    if op_dtype != input_dtype:
        input = input.to(op_dtype)
    y = torch.softmax(input, dim=-1)
    if y.dtype != input_dtype:
        y = y.to(input_dtype)
    return y

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
        self.performer_nb_features = int(
            self.attention_head_size * math.log(self.attention_head_size) / self.pconfig.performer_nb_factor
        )
        self.performer = FastAttention(
            dim_heads = self.attention_head_size,
            nb_features = self.performer_nb_features,
            causal=self.pconfig.causal,
        )
        self.performer_proj_updater = ProjectionUpdater(
            self.performer,
            1000,
        )
        if not self.pconfig.causal:
            performer_value_hidden_size = self.attention_head_size*3
        else:
            performer_value_hidden_size = self.attention_head_size*2
        self.attention_predictor_enc = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(performer_value_hidden_size, self.attention_head_size*2),
            nn.LayerNorm(self.attention_head_size*2),
            nn.GELU(),
        )
        self.attention_predictor_dec_row = nn.Sequential(
            nn.Linear(self.attention_head_size*2, self.pconfig.attention_predictor_length),
        )
        padding_mode = 'reflect'
        # self.attention_predictor_cnn = KeepRes(
        #     # NOTE if we use pixelshuffle outch should be 48
        #     nn.Conv2d(12, 48, 3, padding=1, stride=2, padding_mode=padding_mode),
        #     nn.ReLU(),
        #     Residual(
        #         ResBlock(48, padding=1, lnorm_size=self.pconfig.attention_predictor_length//2, padding_mode=padding_mode),
        #         Residual(KeepRes(
        #             nn.Conv2d(48, 96, 3, padding=1, stride=2, padding_mode=padding_mode),
        #             nn.ReLU(),
        #             ResBlock(96, padding=1, lnorm_size=self.pconfig.attention_predictor_length//4, padding_mode=padding_mode),
        #             nn.Conv2d(96, 48*4, 1, padding=0, stride=1, padding_mode=padding_mode),
        #             nn.PixelShuffle(2),
        #         )),
        #         ResBlock(48, padding=1, lnorm_size=self.pconfig.attention_predictor_length//2, padding_mode=padding_mode),
        #     ),
        #     nn.Conv2d(48, 12*4, 1, padding=0, stride=1, padding_mode=padding_mode),
        #     nn.PixelShuffle(2),
        #     # nn.UpsamplingNearest2d(scale_factor=2),
        #     # UpsampleFP32(2),
        #     # nn.ConvTranspose2d(48, 12, 3, stride=2, padding=1, padding_mode='zeros', output_padding=1),
        #     # nn.ReLU(),
        #     nn.Conv2d(12, 12, 3, padding=1, padding_mode=padding_mode),
        #     nn.LayerNorm(self.pconfig.attention_predictor_length),
        # )
        self.attention_predictor_cnn = KeepRes(
            CausalConv2d(12, 48, 3, padding=1, stride=2, causal=self.pconfig.causal),
            nn.ReLU(),
            ResBlock(48, causal=self.pconfig.causal),
            ResBlock(48, causal=self.pconfig.causal),
            UpsampleFP32(2, torch.float16),
            CausalConv2d(48, 12, 3, padding=1, causal=self.pconfig.causal),
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
        self.norm_performer = nn.LayerNorm(config.hidden_size)
        self.norm_partial = nn.LayerNorm(config.hidden_size)
        self.norm_random = nn.LayerNorm(config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        self.register_buffer('_v_eye', None, persistent=False)
        
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
        if self.pconfig.causal:
            N, H, T1, T2 = attention_mask.shape
            assert T1 == T2
            assert H == 1
            causal_attention_mask = attention_mask
            attention_mask = attention_mask[:, :, :, :1].transpose(-1, -2)
        
        # N, H, T, HID = q.shape
        # return PerlinAttentionOutput(
        #     loss=0,
        #     context_layer=q.transpose(-2, -3).reshape(N, T, H*HID),
        #     partial_attention_probs=None,
        #     estimated_attention_probs=None,
        #     dense_attention_probs=None,
        #     key_for_score=k_for_score,
        # )
        
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
        
        get_bench().register_temp_buffer('q', q)
        get_bench().register_temp_buffer('k', k)
        get_bench().register_temp_buffer('v', v)
        get_bench().register_temp_buffer('attention_mask', attention_mask)
        
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
                    
                    if self._v_eye is None or self._v_eye.shape[-1] != E_N or self._v_eye.dtype != v.dtype:
                        from torch.distributions import Normal
                        def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
                            radius = math.ceil(num_sigmas * sigma)
                            support = torch.arange(-radius, radius + 1, dtype=torch.float)
                            kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
                            # Ensure kernel weights sum to 1, so that image brightness is not altered
                            return kernel.mul_(1 / kernel.sum())
                        
                        def gaussian_filter_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
                            kernel_1d = gaussian_kernel_1d(sigma).to(img.device)  # Create 1D Gaussian kernel
                            
                            padding = len(kernel_1d) // 2  # Ensure that image size does not change
                            img = img.unsqueeze(0).unsqueeze_(0)  # Need 4D data for ``conv2d()``
                            # Convolve along columns and rows
                            img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
                            img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
                            return img.squeeze_(0).squeeze_(0)  # Make 2D again

                        v_for_atten_identity = torch.eye(
                            n=E_N,
                            dtype=v.dtype,
                            device=v.device,
                        )
                        # sig, mu = torch.std_mean(v_for_atten_identity, dim=-1, keepdim=True)
                        # v_for_atten_identity = (v_for_atten_identity - mu) / sig
                        # v_for_atten_identity = gaussian_filter_2d(v_for_atten_identity, 2)
                        # v_for_atten_identity = F.normalize(v_for_atten_identity, dim=-1)
                        
                        v_for_atten_identity = v_for_atten_identity.view(1, 1, E_N, E_N)
                        self._v_eye = v_for_atten_identity
                    else:
                        v_for_atten_identity = self._v_eye
                    
                    # v_for_atten_identity = self.v_eye_learned
                    
                    v_for_atten_identity = v_for_atten_identity.expand(v_for_atten.shape[:2] + (E_N, E_N))
                
                with timer("vmask.grid"):
                    token_index_y = ((zero_one_attention_mask_cumsum - 1.0) / ((zero_one_attention_mask_sum - 1.0).view(N, 1, 1, 1) + 1e-8) * 2 - 1)\
                        .view(N, T, 1, 1)\
                        .expand(N, T, HID, 1)
                    # if self._v_grid_x is None or self._v_grid_x.shape[-2] != HID:
                    #     token_index_x = (torch.arange(HID, device=q.device, dtype=q.dtype) / (HID - 1) * 2 - 1).view(1, 1, HID, 1)
                    #     self._v_grid_x = token_index_x
                    # else:
                    #     token_index_x = self._v_grid_x
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
                
                with timer("vmask.cat_fill"):
                    if not self.pconfig.causal:
                        v_for_atten = torch.cat([
                            v_for_atten_identity, 
                            v_for_atten
                        ], dim=-1)
                    else:
                        #TODO add identity on causal
                        # v_for_atten = torch.cat([
                        #     v_for_atten,
                        #     v_for_atten
                        # ], dim=-1)
                        pass
                    
                    get_bench().register_temp_buffer('v_for_atten', v_for_atten)
                
                    v_for_atten.masked_fill_(attention_mask.transpose(-1, -2) < -1, 0)
                    v.masked_fill_(attention_mask.transpose(-1, -2) < -1, 0)
            
            with timer("performer"):
                if not self.benchmarking:
                    q_type = q_for_atten.dtype
                    if torch.get_autocast_gpu_dtype() in [torch.float16, torch.bfloat16]:
                        PRECISION_PERF = torch.float32 if torch.cuda.is_bf16_supported() else torch.float32
                    else:
                        PRECISION_PERF = torch.float32
                    with torch.autocast('cuda', PRECISION_PERF):
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
                    # with torch.autocast('cuda', torch.float32):
                    #     t_attention_predictor = t_attention_predictor.to(torch.float32)
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
                    estimated_attention_score = softmax_bf16(estimated_attention_score, dim = -1)
                    estimated_attention_score = torch.matmul(
                        estimated_attention_score.view(-1, BOOK_LEN), 
                        self.attention_predictor_comp_codebook
                    )
                    estimated_attention_score = estimated_attention_score.view(N, H, T, -1)
                else:
                    raise Exception()
                get_bench().register_temp_buffer('t_attention_predictor', t_attention_predictor)
            
            # interpolate and convert to probability
            with timer("mask_softmax"):
                T_M = estimated_attention_score.shape[-1]
                estimated_attention_probs = softmax_bf16(estimated_attention_score, -1)
            
            get_bench().register_temp_buffer('estimated_attention_score', estimated_attention_score)
            get_bench().register_temp_buffer('estimated_attention_probs', estimated_attention_probs)
            # print(estimated_attention_probs[0,0,0,:])
            
            # in layerwise, train perlin attention predictor
            def resize_from_m_to_t(x, masked_fill_value, target_width=None):
                N, H, T1, T_M = x.shape
                if target_width is not None:
                    T2 = target_width
                else:
                    T2 = T1
                with timer("resize"):
                    with timer("resize.grid"):
                        if not self.pconfig.causal:
                            token_index_x = zero_one_attention_mask.view(N, 1, T2)
                            if masked_fill_value is not None:
                                # token_index_x = torch.roll(token_index_x, shifts=(1,), dims=(-1)).cumsum(-1) + ((1.0 - zero_one_attention_mask) * 2).view(N, 1, T2)
                                # token_index_x = (token_index_x / ((zero_one_attention_mask.sum(-1) + 2).view(N, 1, 1) + 1e-8) * 2 - 1).expand(N, T1, T2)
                                mask = token_index_x
                                mask_cs = mask.cumsum(-1)
                                token_length = (mask_cs[:, :, -1].unsqueeze(-1) - 1) + 3 * (mask_cs[:, :, -1].unsqueeze(-1)/T_M)
                                token_index_x = torch.clamp(((((mask_cs - 1) + (1 - mask) * 5000)) / (token_length + 1e-8)) * 2 - 1, -1, 1)
                                token_index_x = token_index_x.expand(N, T1, T2)
                                # print(token_index_x[0,0,:], mask_cs[0,0,-1])
                            else:
                                token_index_x = token_index_x.cumsum(-1)
                                token_index_x = (token_index_x / ((zero_one_attention_mask.sum(-1) - 1).view(N, 1, 1) + 1e-8) * 2 - 1).expand(N, T1, T2)
                        else:
                            assert masked_fill_value is not None
                            mask = (causal_attention_mask > -1).float()
                            _N, _H, _TQ, _TK = mask.shape
                            mask_cs = mask.cumsum(-1)
                            # print('mask', mask[0,0,-1,:])
                            # print('mcs', mask_cs[0,0,-1,:])
                            token_length = (mask_cs[:, :, :, -1].unsqueeze(-1) - 1) + 3 * (_TK/T_M)
                            token_index_x = torch.clamp((((mask_cs - 1) + (1 - mask) * (5000  * (_TK/T_M))) / (token_length + 1e-8)) * 2 - 1, -1, 1)
                            assert _H == 1
                            token_index_x = token_index_x[:,0,:,:]
                            # print('tix', token_index_x[0,-1,:])
                        token_index_y = (
                            torch.arange(T1, dtype=torch.long, device=token_index_x.device)\
                                .view(1, T1, 1) / T1 * 2 - 1)\
                                .expand(N, T1, T2) #type: torch.Tensor
                        token_index = torch.cat([
                            token_index_x.unsqueeze(-1),
                            token_index_y.unsqueeze(-1)
                        ], dim=-1)
                    
                    # print('ti', token_index[0,-1,:,:])
                    
                    with timer("resize.sample"):
                        grid_input = F.pad(F.pad(x, pad=(0, 2), value=0), pad=(0, 1), value=masked_fill_value) if masked_fill_value is not None else x
                        if grid_input.dtype != x.dtype:
                            grid_input = grid_input.to(x.dtype)
                        if token_index.dtype != x.dtype:
                            token_index = token_index.to(x.dtype)
                        # print(grid_input.shape, grid_input[0,0,0,-5:])
                        
                        return grid_sample_bf16(
                            input=grid_input,
                            grid=token_index,
                            mode='nearest',
                            align_corners=True,
                            padding_mode='border'
                        )
            
            def resize_from_t_to_m(x, T_M):
                if self.pconfig.causal: raise Exception()
                
                N, H, T1, T2 = x.shape
                with timer("resize"):
                    with timer("resize.grid"):
                        mask = zero_one_attention_mask.view(N, 1, T2)
                        token_length = mask.sum(-1, keepdim=True)
                        token_index_x = ((torch.arange(T_M, device=q.device, dtype=q.dtype).view(1, 1, T_M) / (T_M - 1)) * ((token_length - 1) / (T2 -1)))
                        token_index_x = (token_index_x * 2 - 1).expand(N, T1, T_M)
                        token_index_y = (
                            torch.arange(T1, dtype=token_index_x.dtype, device=token_index_x.device)\
                                .view(1, T1, 1) / T1 * 2 - 1)\
                                .expand(N, T1, T_M) #type: torch.Tensor
                        token_index = torch.cat([
                            token_index_x.unsqueeze(-1), 
                            token_index_y.unsqueeze(-1)
                        ], dim=-1)
                    
                    with timer("resize.sample"):
                        return grid_sample_bf16(
                            input=x,
                            grid=token_index,
                            mode='nearest',
                            align_corners=True,
                            padding_mode='border'
                        )
            
            loss = 0
            if not self.benchmarking:
                T_M = estimated_attention_score.shape[-1]
                # for loss calculation
                estimated_attention_probs_resized = resize_from_m_to_t(estimated_attention_probs, masked_fill_value=0)
                # estimated_attention_probs_resized = resize_from_m_to_t(estimated_attention_probs.fill_(1), masked_fill_value=0)
                # estimated_attention_probs_resized = resize_from_m_to_t(torch.arange(estimated_attention_probs.shape[-1], device=q.device, dtype=q.dtype).view(1, 1, 1, estimated_attention_probs.shape[-1]).expand(estimated_attention_probs.shape) / (estimated_attention_probs.shape[-1]-1), masked_fill_value=0)
                estimated_attention_score_resized = resize_from_m_to_t(estimated_attention_score, masked_fill_value=FP_MIN)
                
                with torch.autocast('cuda', torch.float32):
                    # raise_if_nan(estimated_attention_score_resized)
                    # raise_if_nan(attention_scores_truth)
                    # raise_if_nan(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN))
                    # raise_if_nan(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN))
                    # loss_kl_m = F.kl_div(
                    #     F.log_softmax(estimated_attention_score, dim=-1).view(N, H, T, T_M),
                    #     F.softmax(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1).view(N, H, T, T_M),
                    #     reduction='none'
                    # )
                    # loss_kl_m = loss_kl_m * (attention_mask.transpose(-1, -2) > -1)
                    # loss_kl_m = loss_kl_m.view(N, H, T*T_M).sum(dim=-1, keepdim=True) / (attention_mask > -1).float().sum(dim=-1, keepdim=True)
                    # loss_kl_m = loss_kl_m.mean()
                    # loss_kl_m = loss_kl_m * 0.1
                    # raise_if_nan(loss_kl_m)
                    
                    # loss_mse_m = F.mse_loss(
                    #     softmax_bf16(estimated_attention_score, dim=-1) * (attention_mask.transpose(-1, -2) > -1), 
                    #     softmax_bf16(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1) * (attention_mask.transpose(-1, -2) > -1)
                    # )
                    
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
                        loss_kl_t = F.kl_div(
                            F.log_softmax(estimated_attention_score_resized.masked_fill(causal_attention_mask < -1, FP_MIN), dim=-1).view(-1, estimated_attention_probs_resized.shape[-1]),
                            F.softmax(attention_scores_truth.masked_fill(causal_attention_mask < -1, FP_MIN), dim=-1).view(-1, estimated_attention_probs_resized.shape[-1]),
                            reduction='batchmean',
                        ) * 0.1
                        loss_mse_t = F.mse_loss(
                            softmax_bf16(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN), dim=-1), 
                            softmax_bf16(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1)
                        )
                    
                    raise_if_nan(loss_kl_t)
                    raise_if_nan(loss_mse_t)
                    loss += loss_kl_t + loss_mse_t# + (loss_kl_m + loss_mse_m) * 0.5
                    raise_if_nan(loss)
                    
                get_bench().register_temp_buffer('attention_probs_truth', None, lazy=lambda: F.softmax(attention_scores_truth, dim=-1) * (attention_mask.transpose(-1, -2) > -1))
                get_bench().register_temp_buffer('attention_probs_truth_m', None, lazy=lambda: F.softmax(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1) * (attention_mask.transpose(-1, -2) > -1))
                get_bench().register_temp_buffer('estimated_attention_probs_resized', estimated_attention_probs_resized)
                get_bench().register_temp_buffer('estimated_attention_score_resized', estimated_attention_score_resized)
            
            with timer("mask"):
                estimated_attention_probs = estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1)
                
                T_M = estimated_attention_probs.shape[-1]
                token_length = (attention_mask > -1).long().sum(-1).view(N, -1)
                top_k = min(max(int(round(self.pconfig.k * (T_M / torch.min(token_length).item()))), 1), T_M)
                k_flatten = self.pconfig.k_flatten
                if not k_flatten:
                    with timer("mask.topk"):
                        _, indices = torch.topk(
                            estimated_attention_probs, # estimation gradient is cut here
                            k=top_k, 
                            dim=-1, 
                            sorted=True,
                        )
                    with timer("mask.empty"):
                        partial_attention_mask = torch.empty(
                            (N, H, T, T_M),
                            dtype=q_for_score.dtype,
                            device=q_for_score.device,
                        )
                    with timer("mask.fill"):
                        partial_attention_mask.fill_(FP_MIN)
                    with timer("mask.scatter"):
                        partial_attention_mask.scatter_(dim=-1, index=indices, value=0)
                else:
                    top_k_elems = None
                    per_item_top_k = None 
                    k_flatten_dim = self.pconfig.k_flatten_dim
                    assert k_flatten_dim in ['head', 'batch', 'causal_batch']
                    with timer("mask.view"):
                        masked_estimated_attention_probs = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1))
                        if k_flatten_dim == 'batch':
                            t = masked_estimated_attention_probs.view(N, H*T*T_M)
                            top_k_elems = top_k*T*H
                            per_item_top_k = token_length * H * torch.floor(self.pconfig.k * T_M / token_length)
                        elif k_flatten_dim == 'head':
                            t = masked_estimated_attention_probs.view(N, H, T*T_M)
                            top_k_elems = top_k*T
                            per_item_top_k = token_length * torch.floor(self.pconfig.k * T_M / token_length)
                        elif k_flatten_dim == 'causal_batch':
                            t = masked_estimated_attention_probs.transpose(1, 2).reshape(N, T, H*T_M)
                            top_k_elems = top_k*H
                            # per_item_top_k = (H * self.pconfig.k)
                            per_item_top_k = (H * torch.floor(self.pconfig.k * T_M / token_length)).view(N, 1, 1)
                            # print(t.shape, self.pconfig.k, T, T_M, token_length[0].item(), top_k, top_k_elems, per_item_top_k[0].item())
                        else: raise Exception()
                        get_bench().register_temp_buffer('per_item_top_k', per_item_top_k)
                    with timer("mask.topk"):
                        _, indices = torch.topk(
                            input=t,
                            k=top_k_elems, 
                            dim=-1, 
                            sorted=True #sorted true is important
                        )
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
                        # print(partial_attention_mask[0].view(H, T, T_M)[0])
                    with timer("mask.masked_fill"):
                        # input()
                        # print((token_length * (top_k * H if k_flatten_dim == 'batch' else top_k)).view(-1), token_length.view(-1), top_k, self.pconfig.k, (T_M/token_length).view(-1), per_item_top_k.view(-1))
                        # t_dead_mask = partial_attention_mask >= (token_length * (top_k * H if k_flatten_dim == 'batch' else top_k)) #k is resized
                        if not self.benchmarking:
                            t_dead_mask = partial_attention_mask >= per_item_top_k
                            # partial_attention_mask.fill_(FP_MIN)
                            # partial_attention_mask.masked_fill_(t_alive_mask, value=0)
                            get_bench().register_temp_buffer('t_dead_mask', None, lambda: t_dead_mask.float())
                            partial_attention_mask = t_dead_mask.to(q.dtype) * FP_MIN
                        else:
                            t_alive_mask = partial_attention_mask < per_item_top_k
                            partial_attention_mask = t_alive_mask.float()
                    
                    if k_flatten_dim == 'causal_batch':
                        partial_attention_mask = partial_attention_mask.view(N, T, H, T_M).transpose(1, 2)
                        partial_attention_mask.masked_fill_(
                            mask=attention_mask.transpose(-1, -2) < -1,
                            value=FP_MIN
                        )
                    elif k_flatten_dim in ['batch', 'head']:
                        pass
                    else: raise Exception()
                    partial_attention_mask = partial_attention_mask.view(N, H, T, T_M)
            
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
                        partial_attention_mask = resize_from_m_to_t(partial_attention_mask, FP_MIN)
                        if self.pconfig.causal:
                            partial_attention_mask.masked_fill_(causal_attention_mask < -1, FP_MIN)
                        # print(torch.unique(partial_attention_mask).shape)
                        # print(partial_attention_mask[0,0,0,-10:], attention_mask[0,0,0,-10:])
                        # input()
                else:
                    if self.pconfig.causal: raise Exception()
                    
                    def resize_width(img: torch.Tensor, scale: float):
                        N, H, W = img.shape
                        # img = img.coalesce()
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

                    N, H, T, T_M = partial_attention_mask.shape
                    
                    # original
                    # partial_attention_mask_original = resize_from_m_to_t(partial_attention_mask, FP_MIN if not self.benchmarking else 0).view(N*H, T, T).to_sparse_coo()
                    
                    # optimized
                    partial_attention_mask = partial_attention_mask.reshape(N*H, T, T_M).to_sparse_coo()
                    partial_attention_mask = resize_width(partial_attention_mask, T/T_M)
                
                # with timer("interp.fill"):
                #     partial_attention_mask.masked_fill_(
                #         mask=attention_mask < -1,
                #         value=FP_MIN,
                #     )
                raise_if_nan(partial_attention_mask)
            
            get_bench().register_temp_buffer('partial_attention_mask', partial_attention_mask)
            get_bench().register_temp_buffer('q_for_score', q_for_score)
            get_bench().register_temp_buffer('k_for_score', k_for_score)
            
            # N, H, T, HID = q.shape
            # return PerlinAttentionOutput(
            #     loss=0,
            #     context_layer=q.transpose(-2, -3).reshape(N, T, H*HID),
            #     partial_attention_probs=None,
            #     estimated_attention_probs=None,
            #     dense_attention_probs=None,
            #     key_for_score=k_for_score,
            # )
            
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
                        loss += F.kl_div(
                            F.log_softmax(attention_scores_dense.masked_fill(causal_attention_mask < -1, FP_MIN), dim=-1).view(-1, attention_scores_dense.shape[-1]),
                            F.softmax(attention_scores_truth.masked_fill(causal_attention_mask < -1, FP_MIN), dim=-1).view(-1, attention_scores_dense.shape[-1]),
                            reduction='batchmean',
                        ) * 0.1
                        loss += F.mse_loss(
                            softmax_bf16(attention_scores_dense.masked_fill(causal_attention_mask < -1, FP_MIN), dim=-1), 
                            softmax_bf16(attention_scores_truth.masked_fill(causal_attention_mask < -1, FP_MIN), dim=-1),
                        )
                    get_bench().register_temp_buffer('attention_scores_dense', attention_scores_dense)
                    raise_if_nan(loss)
                    
                    # NOTE HJ `attention_probs_dense` is for visualization, therefore it will not computed on benchmarking mode
                    if not self.pconfig.causal:
                        attention_scores_dense_masked = attention_scores_dense + attention_mask
                    else:
                        attention_scores_dense_masked = attention_scores_dense + causal_attention_mask
                    attention_probs_dense = softmax_bf16(attention_scores_dense_masked, dim=-1)
                    
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
                    partial_attention_probs = softmax_bf16(partial_attention_scores, -1)
                    partial_attention_probs = partial_attention_probs * (partial_attention_mask > -1)
                    get_bench().register_temp_buffer('partial_attention_scores', partial_attention_scores)
                    get_bench().register_temp_buffer('attention_matrix', partial_attention_probs)
                    raise_if_nan(partial_attention_probs)
                    
                    # perform scaling, however this pervent to use spase attention kernel
                    estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                    if self.pconfig.partial_attention_scaler:
                        partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scales[..., 0:1])
                    
                    raise_if_nan(partial_attention_probs)
                    raise_if_nan(v)
                    partial_context_layer = torch.matmul(partial_attention_probs, v)
                    get_bench().register_temp_buffer('partial_context_layer_1', partial_context_layer)
                else:
                    # TODO implement optimized causal kernel
                    if self.pconfig.causal: raise Exception()
                    
                    attention_probs_dense = partial_attention_probs = attention_scores_dense = None
                    partial_context_layer = q_for_score
                    
                    # TODO HJ Apply probs scaler!
                    
                    # NOTE: print avg k per batch
                    # avg_k_per_batch = (((partial_attention_mask.to_dense() > 0).view(N, -1).long().sum(-1) / (attention_mask > -1).long().view(N, -1).sum(-1)).mean() / H).item()
                    # print(metric.update(avg_k_per_batch, name='avgk'))
                    
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
                    if not self.pconfig.causal:
                        average_context_layer = (
                            v *\
                            (attention_mask.transpose(-1, -2) > -1).to(v.dtype) *\
                            resize_from_m_to_t(estimated_attention_probs.mean(-2, keepdim=True), 0, T).transpose(-1, -2)
                        ).sum(-2, keepdim=True).to(v.dtype)
                    else:
                        # TODO imporve this when causal
                        avg_v = v * (attention_mask.transpose(-1, -2) > -1)
                        average_context_layer = avg_v.cumsum(-2) / torch.arange(1, avg_v.shape[-2]+1, device=avg_v.device).view(1, 1, -1, 1)
                        average_context_layer = average_context_layer.to(v.dtype)
                    average_scale = torch.sigmoid(estimated_scales[..., 1:2])
                    partial_context_layer = partial_context_layer * average_scale + (1-average_scale) * average_context_layer
                    get_bench().register_temp_buffer('estimated_scales', estimated_scales)
                    get_bench().register_temp_buffer('average_scale', average_scale)
                    get_bench().register_temp_buffer('estimated_attention_probs_t', None, lazy=lambda: resize_from_m_to_t(estimated_attention_probs.mean(-2, keepdim=True), 0, T).transpose(-1, -2))
                    get_bench().register_temp_buffer('average_context_layer', average_context_layer)
                    get_bench().register_temp_buffer('partial_context_layer_2', partial_context_layer)
            
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
            get_bench().register_temp_buffer('partial_context_layer', partial_context_layer)
            
            return PerlinAttentionOutput(
                loss=loss,
                context_layer=partial_context_layer,
                partial_attention_probs=partial_attention_probs,
                estimated_attention_probs=estimated_attention_probs_for_output,
                dense_attention_probs=attention_probs_dense,
                key_for_score=k_for_score,
            )
