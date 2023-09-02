# this test script only checks whether the outputs(context layer, attention) satisfy the causality,
# without checking the internal implementation (trusting the author)

import argparse
import json
import os
import warnings
from typing import Optional
from ....models.perlin_opt.perlin_opt import OPTAttention

os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import torch
from ....utils import seed

from torch import nn

bool2int = lambda x: 1 if x else 0

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

INDEX_LAYER = 0
MAX_SEQ_LEN = 2048

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

from ....models.perlin_attention.self_attention import get_default_config

def main(baseline, canary, n, h, t_dst, hid, canary_i):
    print(f'\033[93m=====================\033[0m')
    print(f'\033[93mbaseline : {baseline}\033[0m')
    print(f'\033[93mcanary : {canary}\033[0m')
    print(f'\033[93mN : {n}\033[0m')
    print(f'\033[93mH : {h}\033[0m')
    print(f'\033[93mT_DST : {t_dst}\033[0m')
    print(f'\033[93mHID : {hid}\033[0m')
    if canary:
        print(f'\033[93mCANARY_I {canary_i}\033[0m')
    
    print(f'\033[93m=====================\033[0m')

    VIEW_TABLE_2 = True # TODO change to False

    seed()

    ### attention
    from transformers import OPTConfig
    config = OPTConfig()
    config.num_attention_heads = h
    config.hidden_size = h*hid
    config.word_embed_proj_dim = h*hid
    opt_attention = OPTAttention(
        embed_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dropout=config.attention_dropout,
        is_decoder=True,
        bias=config.enable_bias,
    )
    
    opt_attention.attention_method = baseline
    # attention.to(DEVICE)

    #### settings
    hidden_states = torch.randn(n, t_dst, h*hid).to(DEVICE)
    scaling = hid**-0.5

    q_proj = nn.Linear(h*hid, h*hid, bias=True).to(DEVICE)
    k_proj = nn.Linear(h*hid, h*hid, bias=True).to(DEVICE)
    v_proj = nn.Linear(h*hid, h*hid, bias=True).to(DEVICE)
    
    # N, 1, T, T
    causal_attention_mask = _prepare_decoder_attention_mask(
        attention_mask=torch.ones(n, t_dst), 
        input_shape=(n, t_dst),
        device=DEVICE,
        past_key_values_length = 0
    )

    ### truth
    q = (q_proj(hidden_states) * scaling).view(n, t_dst, h, hid).transpose(1,2).contiguous().view(n*h, -1, hid)
    k = k_proj(hidden_states).view(n, -1, h, hid).transpose(1,2).contiguous().view(n*h, -1, hid)
    v = v_proj(hidden_states).view(n, -1, h, hid).transpose(1,2).contiguous().view(n*h, -1, hid)
    
    context_layer_t, attention_probs_t = opt_attention.attention(
        q=q,
        k=k,
        v=v,
        attention_mask=causal_attention_mask,
        last_state=None
    )

    output_truth = {}
    
    output_truth['context_layer'] = context_layer_t
    output_truth['attention_probs'] = attention_probs_t

    #### canary
    seed()

    hidden_states_canary = hidden_states.clone()
    if canary:
        hidden_states_canary[:,canary_i,:] = 300000

    # TODO check whether it's same when hidden_states is in
    q_c = (q_proj(hidden_states_canary) * scaling).view(n, t_dst, h, hid).transpose(1,2).contiguous().view(n*h, -1, hid)
    k_c = k_proj(hidden_states_canary).view(n, -1, h, hid).transpose(1,2).contiguous().view(n*h, -1, hid)
    v_c = v_proj(hidden_states_canary).view(n, -1, h, hid).transpose(1,2).contiguous().view(n*h, -1, hid)
    
    if not canary:
        assert (~torch.eq(q, q_c)).sum().item()==0
        assert (~torch.eq(k, k_c)).sum().item()==0
        assert (~torch.eq(v, v_c)).sum().item()==0

    context_layer_c, attention_probs_c = opt_attention.attention(
        q=q_c,
        k=k_c,
        v=v_c,
        attention_mask=causal_attention_mask,
        last_state=None
    )

    output_canary = {}

    output_canary['context_layer'] = context_layer_c
    output_canary['attention_probs'] = attention_probs_c

    assert output_truth.keys() == output_canary.keys()

    CHECK_INDEX = [0, 1, 2, canary_i-1, canary_i, canary_i+1, canary_i+2, -3, -2, -1]
    JUST_WIDTH = 12
    
    print(f'\n### 1.1 : Attention Output (for all index before {canary_i})')
    # print(f'   {"ERROR=(x-y).abs().sum().log10()".ljust(JUST_WIDTH*3)} | INDEX: {",".join([str(i).rjust(JUST_WIDTH) for i in CHECK_INDEX])}')
    
    total_bugs = 0
    type2_bug = {}
    for k in output_truth.keys():
        truth = output_truth[k]
        mine = output_canary[k]
        losses = []
        for idx in range(canary_i):
            loss = 0
            def error(x, y):
                x = x.to(torch.float64)
                y = y.to(torch.float64)
                return (x - y).abs().sum().log10()
            loss = error(truth[...,idx,:], mine[...,idx,:]).item()
            losses.append(loss)
        def deco_error(str, e):
            if e < -3:
                return f"\033[92m{str}\033[0m" # bright green
            return f"\033[91m{str}\033[0m" # bright red
        for i, loss in enumerate(losses):
            if loss>=-3:
                if k in type2_bug:
                    type2_bug[k][0] += 1
                    if type2_bug[k][2]<loss:
                        type2_bug[k][2] = loss
                else:
                    type2_bug['N']=n
                    type2_bug['H']=h
                    type2_bug['T_DST']=t_dst
                    type2_bug['HID']=hid
                    type2_bug[k] = [1, len(losses), loss]
                # print(f'k | i | loss : {k} {i} \033[91m{loss}\033[0m')
                total_bugs +=1
    if total_bugs == 0:
        print('\033[92mNO BUG EXISTS!\033[0m\n')
    else:
        VIEW_TABLE_2 = True
        print(f"\033[91m{type2_bug}\033[0m")
        os.makedirs(f'./saves/tests/test_{baseline}_opt_causal/canary{bool2int(canary)}', exist_ok=True)
        with open(f'./saves/tests/test_{baseline}_opt_causal/canary{bool2int(canary)}/type2.txt','a') as file:
            file.write(json.dumps(type2_bug)+'\n')
    
    if VIEW_TABLE_2:    
        print('\n### 2.2 : Attention Output')
        print(f'   {"ERROR=(x-y).abs().sum().log10()".ljust(JUST_WIDTH*3)} | INDEX: {",".join([str(i).rjust(JUST_WIDTH) for i in CHECK_INDEX])}')
        
        for k in output_truth.keys():
            truth = output_truth[k]
            mine = output_canary[k]
            losses = []
            for idx in CHECK_INDEX:
                loss = 0
                def error(x, y):
                    x = x.to(torch.float64)
                    y = y.to(torch.float64)
                    return (x - y).abs().sum().log10()
                loss = error(truth[...,idx,:], mine[...,idx,:]).item()
                losses.append(loss)
            def deco_error(str, e):
                if e < -3:
                    return f"\033[92m{str}\033[0m" # bright green
                return f"\033[91m{str}\033[0m" # bright red
            print(f' - {k.ljust(JUST_WIDTH*3)} | ERROR: {",".join([deco_error(f"{loss:.4f}".rjust(JUST_WIDTH), loss) for loss in losses])}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='performer')
    parser.add_argument('--canary', action='store_true', default=False)

    args = parser.parse_args()
    baseline = args.baseline
    canary = args.canary
    
    os.makedirs(f'./saves/tests/test_{baseline}_opt_causal/canary{bool2int(canary)}', exist_ok=True)
    with open(f'./saves/tests/test_{baseline}_opt_causal/canary{bool2int(canary)}/type1.txt','w') as file:
        pass
    with open(f'./saves/tests/test_{baseline}_opt_causal/canary{bool2int(canary)}/type2.txt','w') as file:
        pass

    ### argument
    N = [4, 8, 16] #  2, 8, 16, 
    H = [1, 7, 12, 16, 19, 25, 32, 64] # 1, 7
    T_DST = [2048] # 2048
    HID = [32, 64, ]

    ### pconfig
    pconfig = get_default_config()
    pconfig.causal = True
    pconfig.use_cache = False

    from itertools import product
    list = list(product(N, H, T_DST, HID))

    for l in list:
        main(baseline=baseline,
            canary=canary,
                n=l[0],
                h=l[1],
                t_dst=l[2],
                hid=l[3],
                canary_i=l[2]//2)