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
from ...utils import get_bench
from xformers.components.attention.core import scaled_dot_product_attention, SparseCS

timer = lambda name: get_bench().region(name)

T_MASK = None
T_EYE = None

def interpolate(x: torch.Tensor, size, interp_mode: str = None):
    interp_mode = ('bilinear' if size[-1] >= x.shape[-1] else 'area') if interp_mode is None else interp_mode
    
    if torch.get_autocast_gpu_dtype() == torch.bfloat16: # F interpolate is not supported on bf16
        original_dtype = x.dtype
        with torch.autocast('cuda', torch.float16):
            if x.dtype != torch.float16:
                x = x.to(torch.float16)
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
            causal=False,
            # no_projection=True,
        )
        self.performer_proj_updater = ProjectionUpdater(
            self.performer, 
            1000,
        )
        self.attention_predictor_enc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.attention_head_size*2, self.attention_head_size*2),
            nn.LayerNorm(self.attention_head_size*2),
            nn.GELU(),
        )
        self.attention_predictor_dec_row = nn.Sequential(
            nn.Linear(self.attention_head_size*2, self.pconfig.attention_predictor_length),
        )
        self.attention_predictor_dec_scaler = nn.Sequential(
            nn.Linear(self.attention_head_size*2, 1),
        )
        #-- compressed predictor
        self.attention_predictor_comp_length = \
            self.pconfig.attention_predictor_comp_patch_count * self.pconfig.attention_predictor_comp_patch_size
        self.attention_predictor_comp_codebook = nn.Parameter(
            torch.randn((self.pconfig.attention_predictor_comp_book_size, self.pconfig.attention_predictor_comp_patch_size))
        )
        self.attention_predictor_comp_enc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.all_head_size*2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        self.attention_predictor_comp_dec_row = nn.Sequential(
            nn.Linear(
                config.hidden_size, 
                self.num_attention_heads\
                    * self.pconfig.attention_predictor_comp_book_size\
                    * self.pconfig.attention_predictor_comp_patch_count
            ),
        )
        #-- TODO VQVAE
        
        #- output
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.all_head_size*2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.out_random_lookup = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.all_head_size*3, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.norm_performer = nn.LayerNorm(config.hidden_size)
        self.norm_partial = nn.LayerNorm(config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # from xformers.components.attention import ScaledDotProduct
        # self.xformer_scaled_dot_product = ScaledDotProduct(
        #     dropout = 0.0,
        #     causal = False,
        # )
    
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
        with timer("perlin"):
            # if self.benchmarking:
            #     return PerlinAttentionOutput(
            #         loss=None,
            #         context_layer=context_layer_truth,
            #         dense_attention_probs=None,
            #         estimated_attention_probs=None,
            #         key_for_score=None,
            #         partial_attention_probs=None
            #     )
            
            N, H, T, HID = q.shape
            with timer("vmask"):
                # v_mask = (attention_mask.transpose(-1, -2) > -1)
                # v = v # * v_mask
                # v_for_atten = v_for_atten * v_mask
                global T_EYE
                if T_EYE is None:
                    T_EYE = torch.eye(
                        128, 
                        dtype=v_for_atten.dtype, 
                        device=v_for_atten.device
                    ).view(1, 1, 128, 128)
                v_for_atten = interpolate(
                    x=T_EYE,
                    size=v_for_atten.shape[-2:],
                    interp_mode='nearest'
                ).expand(v_for_atten.shape).contiguous()
                
                v_for_atten.masked_fill_(attention_mask.transpose(-1, -2) < -1, 0)
            
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
            
            with timer("performer_value"):
                performer_value = torch.cat([
                    performer_context_layer, 
                    v
                ], dim=-1).detach()
            N, H, T, HID = performer_value.shape
            # (N, H, T, P)
            
            # estimate attention scores
            with timer("predictor"):
                if self.pconfig.attention_predictor_method == 'mlp':
                    t_attention_predictor = self.attention_predictor_enc(performer_value)
                    estimated_attention_score = self.attention_predictor_dec_row(t_attention_predictor) # type: torch.Tensor
                elif self.pconfig.attention_predictor_method == 'comp':
                    raise Exception()
                    warnings.warn('attention prediction method is compressed one.')
                    t_attention_predictor = self.attention_predictor_comp_enc(performer_value.permute(0,2,1,3).reshape(N, T, H*HID))
                    estimated_attention_score = self.attention_predictor_comp_dec_row(t_attention_predictor)\
                        .view(N, T, H, -1).permute(0, 2, 1, 3)
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
                resized_attention_mask = interpolate(
                    x=attention_mask, 
                    size=(1, estimated_attention_score.shape[-1]), 
                    interp_mode='nearest',
                )
                resized_attention_mask_binary = resized_attention_mask < -1
                # resized_attention_mask = (resized_attention_mask < -1) * -10000
                if self.benchmarking:
                    estimated_attention_score = estimated_attention_score.masked_fill_(
                        mask=resized_attention_mask_binary,
                        value=-10000
                    )
                else:
                    estimated_attention_score_unmasked = estimated_attention_score
                    estimated_attention_score = estimated_attention_score.masked_fill(
                        mask=resized_attention_mask_binary,
                        value=-10000
                    )
                estimated_attention_probs = torch.softmax(estimated_attention_score, -1)
            
            # in layerwise, train perlin attention predictor
            if not self.benchmarking:
                estimated_attention_probs_resized = interpolate(
                    x=estimated_attention_probs, 
                    size=(T, T), 
                    interp_mode='nearest'
                )
                estimated_attention_score_resized = interpolate(
                    x=estimated_attention_score_unmasked, 
                    size=(T, T), 
                    interp_mode='nearest'
                )
                
                with torch.autocast('cuda', torch.float32):
                    loss_kl = kl_div_attention(
                        F.log_softmax(estimated_attention_score_resized + attention_mask, dim=-1),
                        F.softmax(attention_scores_truth + attention_mask, dim=-1),
                        attention_mask,
                    ) * 0.25
                    loss_mse = F.mse_loss(
                        estimated_attention_score_resized.masked_fill(attention_mask < -1, 0), 
                        attention_scores_truth.masked_fill(attention_mask < -1, 0)
                    )
                    loss = loss_kl + loss_mse
            else:
                loss = 0
            
            with timer("mask"):
                T_M = estimated_attention_probs.shape[-1]
                k = min(max(int(round(self.pconfig.k * (T_M / T))), 1), T_M)
                k_flatten = self.pconfig.k_flatten
                if not k_flatten:
                    raise Exception("not precise")
                    with timer("mask.topk"):
                        _, indices = torch.topk(
                            estimated_attention_probs, # estimation gradient is cut here
                            k=k, dim=-1, sorted=True
                        )
                    with timer("mask.empty"):
                        partial_attention_mask = torch.empty(
                            (N, H, T, T_M),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                    with timer("mask.fill"):
                        partial_attention_mask.fill_(-10000)
                    with timer("mask.scatter"):
                        partial_attention_mask.scatter_(dim=-1, index=indices, value=0)
                else:
                    k_flatten_dim = self.pconfig.k_flatten_dim
                    assert k_flatten_dim in ['head', 'batch']
                    with timer("mask.view"):
                        t = (estimated_attention_probs * (attention_mask.transpose(-1, -2) > -1)).view(N, H*T*T_M)
                    with timer("mask.topk"):
                        _, indices = torch.topk(
                            input=t,
                            k=k*T*H if k_flatten_dim == 'batch' else k*T, 
                            dim=-1, 
                            sorted=True #sorted true is important
                        )
                    with timer("mask.empty"):
                        partial_attention_mask = torch.empty(
                            t.shape, 
                            dtype=attention_mask.dtype, 
                            device=attention_mask.device
                        )
                    with timer("mask.fill"):
                        partial_attention_mask.fill_(t.shape[-1])
                    with timer("mask.scatter"):
                        partial_attention_mask.scatter_(
                            dim=-1,
                            index=indices,
                            src=torch.arange(
                                k*T*H if k_flatten_dim == 'batch' else k*T, 
                                device=attention_mask.device, 
                                dtype=attention_mask.dtype
                            )\
                                .view((1, -1) if k_flatten_dim == 'batch' else (1, 1, -1))\
                                .expand(indices.shape)
                        )
                        # print((partial_attention_mask[0,0].view(T, -1)[:15, :]).long())
                    with timer("mask.masked_fill"):
                        # print(partial_attention_mask[0,0,:].view(T, T_M).long(), ((resized_attention_mask > -1).float().sum(-1).view(N, 1, -1) * k)[0, 0], k)
                        t_alive_mask = partial_attention_mask < ((attention_mask > -1).float().sum(-1).view(N, -1) * (k * H if k_flatten_dim == 'batch' else k)) #k is resized
                        partial_attention_mask.fill_(-10000)
                        partial_attention_mask.masked_fill_(t_alive_mask, value=0)
                        # print(partial_attention_mask[0,0].view(T, T_M))
                    partial_attention_mask = partial_attention_mask.view(N, H, T, T_M)
            
            with timer("interp"):
                # print((partial_attention_mask[0,0][:15, :] > -1).long())
                # print(
                #     (partial_attention_mask > -1).long().sum(-1).sum(-1)[:,0].view(-1),
                #     (attention_mask > -1).long().sum(-1).view(-1),
                #     (partial_attention_mask > -1).long().sum(-1).sum(-1)[:,0].view(-1) / (attention_mask > -1).long().sum(-1).view(-1),
                #     k, T, T_M
                # )
                partial_attention_mask = interpolate(partial_attention_mask, (T, T), interp_mode='nearest')
                # print((partial_attention_mask[0,0][:15, :15]).float())
                # partial_attention_mask.masked_fill_(partial_attention_mask < (-10000 * 0.5), torch.finfo(partial_attention_mask.dtype).min)
            
            with timer("attention"):
                if not self.benchmarking:
                    # print((partial_attention_mask[0,0][:15, :15] > -1).long())
                    # print(
                    #     (partial_attention_mask > -1).long().sum(-1).sum(-1)[:,0].view(-1),
                    #     (attention_mask > -1).long().sum(-1).view(-1),
                    #     (partial_attention_mask > -1).long().sum(-1).sum(-1)[:,0].view(-1) / (attention_mask > -1).long().sum(-1).view(-1),
                    #     k, T, T_M
                    # )
                    # start of attention mechanism
                    attention_scores_dense = torch.matmul(q_for_score, k_for_score.transpose(-1, -2))
                    attention_scores_dense = attention_scores_dense / math.sqrt(self.attention_head_size)
                    # if attention_mask is not None:
                    #     attention_scores_dense = attention_scores_dense + attention_mask
                    loss += F.mse_loss(
                        attention_scores_dense.masked_fill(attention_mask < -1, 0), 
                        attention_scores_truth.masked_fill(attention_mask < -1, 0),
                    ) * 0.5
                    attention_probs_dense = torch.softmax(attention_scores_dense, dim=-1)
                    
                    partial_attention_scores = attention_scores_dense + partial_attention_mask
                    # print(partial_attention_scores)
                    partial_attention_probs = torch.softmax(partial_attention_scores, -1)
                    
                    # perform scaling, however this pervent to use spase attention kernel
                    if self.pconfig.partial_attention_scaler:
                        estimated_scale = self.attention_predictor_dec_scaler(t_attention_predictor)
                        partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scale)
                    
                    partial_context_layer = torch.matmul(partial_attention_probs, v)
                else:
                    attention_probs_dense = partial_attention_probs = attention_scores_dense = None
                    partial_context_layer = q_for_score
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

            with timer("context_permute"):
                partial_context_layer = partial_context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = partial_context_layer.size()[:-2] + (self.all_head_size,)
                partial_context_layer = partial_context_layer.view(new_context_layer_shape)
                performer_context_layer = performer_context_layer.permute(0, 2, 1, 3).contiguous()
                performer_context_layer = performer_context_layer.view(new_context_layer_shape)
            
            with timer("out"):
                if not self.pconfig.random_lookup:
                    partial_context_layer = \
                        self.norm_partial(partial_context_layer) +\
                        partial_context_layer
                        # self.norm_performer(performer_context_layer) +\
                else:
                    partial_context_layer = self.out_random_lookup(torch.cat([
                        partial_context_layer, 
                        performer_context_layer,
                        random_context_layer,
                    ], dim=-1)) + partial_context_layer
            
            with timer("norm"):
                # partial_context_layer = self.norm(partial_context_layer)
                pass
            
            # in layerwise train only norm
            if not self.benchmarking:
                loss += F.mse_loss(context_layer_truth, partial_context_layer)

            return PerlinAttentionOutput(
                loss=loss,
                context_layer=partial_context_layer,
                partial_attention_probs=partial_attention_probs,
                estimated_attention_probs=estimated_attention_probs if self.benchmarking else estimated_attention_probs_resized,
                dense_attention_probs=attention_probs_dense,
                key_for_score=k_for_score,
            )