import os, tqdm, gc
import warnings
from typing import Optional
import flax

os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import numpy as np
import torch
from .common_opt import init
from ...models import perlin_attention
from ...utils import get_bench, seed, strify
import torch.nn.functional as F

from torch import nn


torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

TRACKING_BUFFERS = [
    'q',
    'k',
    'v_for_atten',
    'performer_context_layer',
    'performer_value',
    't_attention_predictor',
    'estimated_attention_score_dec_row',
    'estimated_attention_score',
    'estimated_attention_probs',
    'partial_attention_mask_before_interp',
    'partial_attention_mask',
    'partial_attention_scores',
    'estimated_scales',
    'average_scale',
    'average_context_layer',
    'partial_context_layer_sparse',
    'normalized_partial_context_layer',
    'partial_context_layer',
]


INDEX_LAYER = 0
MAX_SEQ_LEN = 2048

i = 0 
N = [1, ][i]
H = [12, ][i]
T_DST = [20, ][i]
HID = [64, ][i]

DTYPE = torch.float32 # check
DEVICE = 0

warnings.warn(f'DTYPE setteled as {DTYPE}')

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device) # JIN WHY differs from FP_MIN
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(attention_mask, input_shape, device, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                dtype = DTYPE,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, dtype=DTYPE, tgt_len=input_shape[-1]).to(
                device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

from ...models.perlin_attention.self_attention import PerlinSelfAttention, get_default_config
from transformers.models.bert.configuration_bert import BertConfig


def main():
    seed()
    use_cache = True
    bench = get_bench()
    bench.disabled = False
    bench.activate_temp_buffers = True

    pconfig = get_default_config()
    pconfig.causal = True
    pconfig.use_cache = False

    perlin_self_attention = PerlinSelfAttention(
        BertConfig(hidden_size=H*HID, num_attention_heads=H), # WHY
        perlin_config=pconfig,
    )

    perlin_self_attention.to(DEVICE)
    
    # q, k, v
    # attention_mask, attention_scores_truth, context_layer_truth, last_state
    hidden_states = torch.randn(N, T_DST, H*HID).to(DEVICE)
    scaling = HID**-0.5

    q_proj = nn.Linear(H*HID, H*HID, bias=True).to(DEVICE)
    k_proj = nn.Linear(H*HID, H*HID, bias=True).to(DEVICE)
    v_proj = nn.Linear(H*HID, H*HID, bias=True).to(DEVICE)
    
    # N, 1, T, T
    causal_attention_mask = _prepare_decoder_attention_mask(
        attention_mask=torch.ones(N, T_DST), 
        input_shape=(N, T_DST),
        device=DEVICE,
        past_key_values_length = 0
    )
    causal_attention_mask_c = causal_attention_mask.clone()

    attention_scores_truth = torch.randn(N, H, T_DST, T_DST).to(DEVICE)
    attention_scores_truth_c = attention_scores_truth.clone()
    context_layer_truth = torch.randn(N*H, T_DST, HID)\
                .view(N, H, T_DST, HID)\
                .transpose(1, 2)\
                .reshape(N, T_DST, H*HID)\
                .to(DEVICE)
    context_layer_truth_c = context_layer_truth.clone()

    q = (q_proj(hidden_states) * scaling).view(N, T_DST, H, HID).transpose(1,2).contiguous().view(N*H, -1, HID).view(N, H, -1, HID)
    k = k_proj(hidden_states).view(N, -1, H, HID).transpose(1,2).contiguous().view(N*H, -1, HID).view(N, H, -1, HID)
    v = v_proj(hidden_states).view(N, -1, H, HID).transpose(1,2).contiguous().view(N*H, -1, HID).view(N, H, -1, HID)
    '''
    PerlinAttentionOutput]
    loss: torch.Tensor
    context_layer: torch.Tensor
    partial_attention_probs: torch.Tensor
    partial_attention_mask: torch.Tensor
    estimated_attention_probs_m: torch.Tensor
    estimated_attention_probs: torch.Tensor
    dense_attention_probs: torch.Tensor
    key_for_score: torch.Tensor
    state: PerlinAttentionState
    '''

    output_truth = {}

    output_t = perlin_self_attention(
        query = q_proj,
        key = k_proj,
        value = v_proj,
        hidden_states = None,
        query_layer = q,
        key_layer = k,
        value_layer = v,
        attention_mask = causal_attention_mask,
        attention_scores_truth = attention_scores_truth,
        context_layer_truth = context_layer_truth,
        last_state = None, # check - for use_cache
    ) #type: PerlinAttentionOutput

    buffers_truth = {}
    for name in TRACKING_BUFFERS:
        # sample only first layer
        if name in bench.buffers:
            buffers_truth[name] = bench.get_temp_buffer(name, index=INDEX_LAYER)

    bench.reset_temp_buffers()

    for name, value in vars(output_t).items():
        output_truth[name] = value

    hidden_states_canary = hidden_states.clone()
    canary_i = T_DST//2

    canary = True
    if canary:
        hidden_states_canary[:,canary_i,:] = 300000

    # TODO check whether it's same when hidden_states is in
    q_c = (q_proj(hidden_states_canary) * scaling).view(N, T_DST, H, HID).transpose(1,2).contiguous().view(N*H, -1, HID).view(N, H, -1, HID)
    k_c = k_proj(hidden_states_canary).view(N, -1, H, HID).transpose(1,2).contiguous().view(N*H, -1, HID).view(N, H, -1, HID)
    v_c = v_proj(hidden_states_canary).view(N, -1, H, HID).transpose(1,2).contiguous().view(N*H, -1, HID).view(N, H, -1, HID)
    
    if not canary:
        assert (~torch.eq(q, q_c)).sum().item()==0
        assert (~torch.eq(k, k_c)).sum().item()==0
        assert (~torch.eq(v, v_c)).sum().item()==0

    output_canary = {}

    output_c = perlin_self_attention(
        query = q_proj,
        key = k_proj,
        value = v_proj,
        hidden_states = None,
        query_layer = q_c,
        key_layer = k_c,
        value_layer = v_c,
        attention_mask = causal_attention_mask_c,
        attention_scores_truth = attention_scores_truth_c,
        context_layer_truth = context_layer_truth_c,
        last_state = None, # check - for use_cache
    ) #type: PerlinAttentionOutput

    buffers_canary = {}
    for name in TRACKING_BUFFERS:
        # sample only first layer
        if name in bench.buffers:
            buffers_canary[name] = bench.get_temp_buffer(name, index=INDEX_LAYER)
    
    bench.reset_temp_buffers()

    for name, value in vars(output_c).items():
        output_canary[name] = value
    
    assert output_truth.keys() == output_canary.keys()

    CHECK_INDEX = [0, 1, 2, canary_i-1, canary_i, canary_i+1, canary_i+2, -3, -2, -1]
    JUST_WIDTH = 12
    print(f'\nT_DST : {T_DST}')
    print(f'canary_i : {canary_i}')

    print('\n### 1 : TRACKING_BUFFERS')
    print(f'   {"ERROR=(x-y).abs().sum().log10()".ljust(JUST_WIDTH*3)} | INDEX: {",".join([str(i).rjust(JUST_WIDTH) for i in CHECK_INDEX])}')
    for name in TRACKING_BUFFERS:
        if name in buffers_truth and name in buffers_canary:
            truth = buffers_truth[name]
            mine = buffers_canary[name]
            losses = []
            for idx in CHECK_INDEX:
                def error(x, y):
                    x = x.to(torch.float64)
                    y = y.to(torch.float64)
                    return (x - y).abs().sum().log10()
                loss = error(truth[...,idx,:], mine[...,idx,:]).item()
                # if name=='partial_attention_mask_before_interp':breakpoint()
                # if name=='partial_attention_mask':breakpoint()
                losses.append(loss)
            def deco_error(str, e):
                if e < -3:
                    return f"\033[92m{str}\033[0m"
                return f"\033[91m{str}\033[0m"
            print(f' - {name.ljust(JUST_WIDTH*3)} | ERROR: {",".join([deco_error(f"{loss:.4f}".rjust(JUST_WIDTH), loss) for loss in losses])}')
    
    print('\n### 2 : PerlinAttentionOutput')
    print(f'   {"ERROR=(x-y).abs().sum().log10()".ljust(JUST_WIDTH*3)} | INDEX: {",".join([str(i).rjust(JUST_WIDTH) for i in CHECK_INDEX])}')
    
    for k in output_truth.keys():
        if k=='state':
            continue
        truth = output_truth[k]
        mine = output_canary[k]
        losses = []
        for idx in CHECK_INDEX:
            loss = 0
            def error(x, y):
                x = x.to(torch.float64)
                y = y.to(torch.float64)
                return (x - y).abs().sum().log10()
            if k=='loss':
                loss = error(truth, mine).item()
            else:
                loss = error(truth[...,idx,:], mine[...,idx,:]).item()
            # if k=='partial_attention_mask':breakpoint()
            losses.append(loss)
        def deco_error(str, e):
            if e < -3:
                return f"\033[92m{str}\033[0m" # bright green
            return f"\033[91m{str}\033[0m" # bright red
        print(f' - {k.ljust(JUST_WIDTH*3)} | ERROR: {",".join([deco_error(f"{loss:.4f}".rjust(JUST_WIDTH), loss) for loss in losses])}')

if __name__ == '__main__':
    main()