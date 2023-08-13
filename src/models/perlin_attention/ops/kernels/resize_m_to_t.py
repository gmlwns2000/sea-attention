import torch, math
import torch.nn.functional as F

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

def resize_from_m_to_t(x, masked_fill_value, causal_attention_mask, target_width=None, output_dtype=None, training=False):
    N, H, T1, T_M = x.shape
    if target_width is not None:
        T2 = target_width
    else:
        T2 = T1
    
    assert masked_fill_value is not None
    mask = (causal_attention_mask > -1).float()
    _N, _H, _TQ, _TK = mask.shape
    mask_cs = mask.cumsum(-1)
    token_length = mask_cs[:, :, :, -1].unsqueeze(-1) 
    if training:
        mask_cs = torch.clamp(mask_cs + (torch.rand_like(mask_cs) * 4 - 2), torch.min(mask_cs), torch.max(mask_cs))
    token_index_x = torch.floor(((mask_cs - 1) + 0.5) / token_length * T_M - 1e-4) + 0.5
    # print(token_index_x[0, 0, 5], (((mask_cs - 1) + 0.5) / token_length * T_M - 1e-4)[0, 0, 5], token_length[0, 0, 5])
    token_index_x = (token_index_x / (T_M + 1)) * 2 - 1 + (1 - mask) * (5000)
    token_index_x = torch.clamp_max(token_index_x, 1)
    # token_index_x = torch.clamp((((mask_cs - 1) + (1 - mask) * (5000)) / (token_length + 1e-8)) * 2 - 1, -1, 1)
    assert _H == 1
    
    print(token_index_x[0, 0, -1])
    
    token_index_x = token_index_x[:,0,:,:].expand(N, T1, T2)
    token_index_y = (
        (torch.arange(T1, dtype=torch.long, device=token_index_x.device) + 0.5)\
            .view(1, T1, 1) / T1 * 2 - 1)\
            .expand(N, T1, T2) #type: torch.Tensor
    
    print(token_index_y[0, -1])
    
    token_index = torch.cat([
        token_index_x.unsqueeze(-1),
        token_index_y.unsqueeze(-1)
    ], dim=-1)
        
    grid_input = F.pad(x, pad=(0, 1), value=masked_fill_value) if masked_fill_value is not None else x

    if grid_input.dtype != x.dtype:
        grid_input = grid_input.to(x.dtype)
    if token_index.dtype != x.dtype:
        token_index = token_index.to(x.dtype)
    
    return grid_sample_bf16(
        input=grid_input,
        grid=token_index,
        mode='nearest',
        align_corners=False,
        padding_mode='border',
        output_dtype=output_dtype,
    )