import torch, math
import os, tqdm, gc
import torch.nn.functional as F
import time
import triton
import triton.language as tl

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

def causal_topk_masking(
    probs, 
    k,
    attention_mask, 
    dst_attention_mask, 
    causal_attention_mask, 
    not_padded=True, 
    k_flatten_dim='causal_batch'
):
    # attention_mask is always for src
    assert k_flatten_dim == 'causal_batch'
    assert not_padded
    
    N, H, T_DST, T_M = probs.shape
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    
    top_k_elems = None
    per_item_top_k = None
    assert k_flatten_dim in ['head', 'batch', 'causal_batch']
        
    masked_estimated_attention_probs = (probs * (dst_attention_mask > -1))
    
    causal_token_length = (causal_attention_mask > -1).long().sum(-1).view(1, 1, T_DST, 1)
    
    t = masked_estimated_attention_probs.transpose(1, 2).reshape(N, T_DST, H*T_M)
    # NOTE consider causal token length
    per_item_top_k = torch.clamp((H * torch.floor(k * T_M / causal_token_length.squeeze(0))).view(1, T_DST, 1), 1, H*T_M)
    
    # NOTE to prevent 0 top-k when large T and small T_m
    per_item_top_k = torch.clamp_min(per_item_top_k, 1)
    
    top_k_elems = min(int(math.ceil(torch.max(per_item_top_k).item())), t.shape[-1])
        
    _, indices = torch.topk(
        input=t,
        k=top_k_elems, 
        dim=-1, 
        sorted=True #sorted true is important
    )
        
    partial_attention_mask = torch.empty(
        t.shape, 
        dtype=torch.long, 
        device=attention_mask.device,
    )
    partial_attention_mask.fill_(t.shape[-1])
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
        
    t_alive_mask = partial_attention_mask < per_item_top_k
    partial_attention_mask = t_alive_mask.float()
    
    partial_attention_mask = partial_attention_mask.view(N, T_DST, H, T_M).transpose(1, 2)
    partial_attention_mask.masked_fill_(
        mask=dst_attention_mask < -1,
        value=FP_MIN
    )
    
    partial_attention_mask = partial_attention_mask.view(N, H, T_DST, T_M)
    
    return partial_attention_mask

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
    token_length = (mask_cs[:, :, :, -1].unsqueeze(-1) - 1) + 3 * math.floor(_TK/T_M)
    if training:
        mask_cs = torch.clamp(mask_cs + (torch.rand_like(mask_cs) * 4 - 2), torch.min(mask_cs), torch.max(mask_cs))
    token_index_x = torch.clamp((((mask_cs - 1) + (1 - mask) * (5000)) / (token_length + 1e-8)) * 2 - 1, -1, 1)
    assert _H == 1
    token_index_x = token_index_x[:,0,:,:]
    token_index_y = (
        torch.arange(T1, dtype=torch.long, device=token_index_x.device)\
            .view(1, T1, 1) / (T1 - 1) * 2 - 1)\
            .expand(N, T1, T2) #type: torch.Tensor
    
    token_index = torch.cat([
        token_index_x.unsqueeze(-1),
        token_index_y.unsqueeze(-1)
    ], dim=-1)
        
    grid_input = F.pad(F.pad(x, pad=(0, 2), value=0), pad=(0, 1), value=masked_fill_value) if masked_fill_value is not None else x

    if grid_input.dtype != x.dtype:
        grid_input = grid_input.to(x.dtype)
    if token_index.dtype != x.dtype:
        token_index = token_index.to(x.dtype)
    
    return grid_sample_bf16(
        input=grid_input,
        grid=token_index,
        mode='nearest',
        align_corners=True,
        padding_mode='border',
        output_dtype=output_dtype,
    )

def scan_col_py(x, original_width, target_width, max_col_z):
    N, A, B = x.shape
    assert target_width.shape == (A,)
    ncols = torch.zeros((N, A), dtype=torch.long)
    col_indices = torch.zeros((N, A, max_col_z))
    for n in range(N): #prange
        for a in range(A): #prange
            last_index = 0
            for b in range(B):
                if x[n, a, b] != 0:
                    n_pixel = 0
                    v = b #x[n, a, b]
                    scale = target_width[a] / original_width
                    v_start = torch.round(v * scale)
                    v_end = torch.round((v+1) * scale)
                    n_pixel = v_end - v_start
                    n_pixel = int(n_pixel.item())
                    for i in range(n_pixel):
                        col_indices[n, a, last_index+i] = (v_start + i)
                    last_index += n_pixel
            ncols[n, a] = last_index
    return ncols, col_indices

@triton.jit
def __scan_col_compute(
    X,
    stride_xn, stride_xa, stride_xb,
    N, A, B: tl.constexpr, BLOCK_A: tl.constexpr,
    SCALE,
    stride_scale,
    NCOLS,
    stride_ncolsn, stride_ncolsa,
    COL_INDICES,
    stride_coln, stride_cola, stride_colz,
    MAX_Z: tl.constexpr,
    MAX_INTERP: tl.constexpr,
):
    # for n in range(N): #prange
    #     for a in range(A): #prange
    #         last_index = 0
    #         for b in range(B):
    #             if x[n, a, b] != 0:
    #                 n_pixel = 0
    #                 v = b #x[n, a, b]
    #                 scale = scales[a]
    #                 v_start = torch.round(v * scale)
    #                 v_end = torch.round((v+1) * scale)
    #                 n_pixel = v_end - v_start
    #                 n_pixel = int(n_pixel.item())
    #                 for i in range(n_pixel):
    #                     col_indices[n, a, last_index+i] = (v_start + i)
    #                 last_index += n_pixel
    #         ncols[n, a] = last_index
    n = tl.program_id(0)
    block_a = tl.program_id(1)
    
    for ia in range(BLOCK_A):
        a = block_a * BLOCK_A + ia
        mask_a = a < A
        
        scales_a = tl.load(SCALE + a*stride_scale, mask=mask_a, other=0)
        
        last_index = int(0)
        for b in range(B):
            x_mask = tl.load(X + n*stride_xn + a*stride_xa + b*stride_xb, mask=mask_a, other=0).to(tl.int32)
            v_start = tl.math.round(b*scales_a)
            v_end = tl.math.round((b+1)*scales_a)
            n_pixel = tl.math.ceil(v_end-v_start).to(tl.int32) * x_mask
            tl.store(
                COL_INDICES \
                    + n*stride_coln \
                    + a*stride_cola \
                    + (tl.arange(0, MAX_INTERP) + last_index.to(tl.int64)) * stride_colz,
                tl.arange(0, MAX_INTERP) + v_start,
                mask=(tl.arange(0, MAX_INTERP) < n_pixel) & mask_a,
            )
            # tl.store(
            #     COL_INDICES \
            #         + n*stride_coln \
            #         + a*stride_cola \
            #         + (tl.arange(0, MAX_INTERP) + last_index.to(tl.int64)) * stride_colz,
            #     1,
            #     mask=(tl.arange(0, MAX_INTERP) < -1),
            # )
            last_index += n_pixel
        
        tl.store(NCOLS + n*stride_ncolsn + a*stride_ncolsa, last_index, mask=mask_a)

def scan_col(x: torch.Tensor, original_width: int, target_width_max: int, target_width: torch.Tensor, max_col_z: int):
    N, A, B = x.shape
    assert target_width.shape == (A,)
    ncols = torch.zeros((N, A), dtype=torch.long, device=x.device)
    col_indices = torch.zeros((N, A, max_col_z), device=x.device, dtype=torch.long)
    scales = target_width / original_width
    
    BLOCK_A = 16
    grid = (N, A//BLOCK_A,)
    __scan_col_compute[grid](
        x, 
        x.stride(0), x.stride(1), x.stride(2),
        N, A, B, BLOCK_A,
        scales, 
        scales.stride(0),
        ncols, 
        ncols.stride(0), ncols.stride(1), 
        col_indices, 
        col_indices.stride(0), col_indices.stride(1), col_indices.stride(2), 
        max_col_z,
        triton.next_power_of_2(int(math.ceil(target_width_max / original_width))),
    )
    
    # truth = scan_col_py(x, original_width=original_width, target_width=target_width, max_col_z=max_col_z)
    
    # print(ncols)
    # print(col_indices[0])
    # print(truth[0], torch.any(truth[0].to(ncols.device) != ncols))
    # print(truth[1][0])
    
    return ncols, col_indices

def compute_cols_py(ncols_cs, col_indices, out_col_indices):
    N, A, _ = col_indices.shape
    for n in range(N):
        for a in range(A):
            out_col_indices[n, ncols_cs[n, a]:ncols_cs[n, a+1]] = col_indices[n, a, :ncols_cs[n, a+1]-ncols_cs[n, a]]

@triton.jit
def __compute_cols_compute(
    NCOLS_CS,
    stride_ncols_cs_n, stride_ncols_cs_a,
    N, A,
    COL_INDICES,
    stride_col_indices_n, stride_col_indices_a, stride_col_indices_mz,
    OUT_COL_INDICES,
    stride_out_col_indices_n, stride_out_col_indices_z,
    MAX_NCOLS:tl.constexpr,
    BLOCK_A:tl.constexpr,
):
    n = tl.program_id(0)
    pid_a = tl.program_id(1)
    
    # a = pid_a * BLOCK_A + tl.arange(BLOCK_A)
    
    #out_col_indices[n, ncols_cs[n, a]:ncols_cs[n, a+1]] = col_indices[n, a, :ncols_cs[n, a+1]-ncols_cs[n, a]]
    for ia in tl.static_range(BLOCK_A):
        a = pid_a * BLOCK_A + ia
        mask_a = a < A
        
        cs_start = tl.load(NCOLS_CS+n*stride_ncols_cs_n+a*stride_ncols_cs_a, mask=mask_a)
        cs_end = tl.load(NCOLS_CS+n*stride_ncols_cs_n+(a + 1)*stride_ncols_cs_a, mask=mask_a)
        cs_len = cs_end - cs_start
        col_indices = tl.load(
            COL_INDICES \
                + n*stride_col_indices_n\
                + a*stride_col_indices_a\
                + tl.arange(0, MAX_NCOLS),
            mask = (tl.arange(0, MAX_NCOLS) < cs_len) & mask_a
        )
        tl.store(
            OUT_COL_INDICES\
                + n*stride_out_col_indices_n\
                + (tl.arange(0, MAX_NCOLS) + cs_start)*stride_out_col_indices_z,
            col_indices,
            mask = (tl.arange(0, MAX_NCOLS) < cs_len) & mask_a
        )

def compact_cols(ncols, col_indices: torch.Tensor):
    N, A = ncols.shape
    N, A, MZ = col_indices.shape
    ncols_cs = F.pad(ncols.view(1, 1, N, A), pad=(1, 0), mode='constant', value=0).view(N, A+1).cumsum(-1)
    z_per_batch = ncols_cs[0,-1]
    out_col_indices = torch.zeros((N, z_per_batch), dtype=torch.long, device=ncols.device) # type: torch.Tensor
    
    # print()
    BLOCK_A = 16
    grid = (N, A//BLOCK_A)
    # print(triton.next_power_of_2(max(1, int(torch.max(ncols).item()))))
    __compute_cols_compute[grid](
        ncols_cs,
        ncols_cs.stride(0), ncols_cs.stride(1),
        N, A,
        col_indices,
        col_indices.stride(0), col_indices.stride(1), col_indices.stride(2),
        out_col_indices,
        out_col_indices.stride(0), out_col_indices.stride(1),
        triton.next_power_of_2(max(1, int(torch.max(ncols).item()))),
        BLOCK_A,
    )
    
    return ncols_cs, out_col_indices

def resize_from_m_to_t_csr(x, masked_fill_value, k, target_width=None, training=False):
    assert not training
    assert masked_fill_value == 0
    N, H, T_DST, T_M = x.shape
    if target_width is not None:
        T_SRC = target_width
    else:
        T_SRC = T_DST
    
    x = x.reshape(N, H*T_DST, T_M)
    
    ncols, _col_indices = scan_col(
        x, 
        original_width=T_M, 
        target_width_max=T_SRC, 
        target_width=(torch.arange(1, T_SRC+1, device=x.device).repeat(H)), 
        max_col_z=H*k
    )
    # print(ncols, _col_indices)
    # print(ncols.shape, _col_indices.shape)
    crows_indices, col_indices = compact_cols(ncols, _col_indices)
    
    # print(crows_indices, col_indices, _col_indices)
    return torch.sparse_csr_tensor(
        crow_indices=crows_indices,
        col_indices=col_indices,
        values=torch.ones(col_indices.shape, device=col_indices.device),
        size=(N, H*T_DST, T_DST),
    )

def test_main():
    from ....utils import seed
    from ....utils.bench import bench

    seed()
    
    # N = 1
    # H = 12
    # T = 8
    # T_M = 4
    # K = 4
    
    N = 1
    H = 12
    T = 1024
    T_M = 128
    K = 128
    
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    device = 0
    
    estimated_scores = torch.randn((N, H, T, T_M), device=device)
    estimated_probs = torch.softmax(estimated_scores, dim=-1)
    causal_attention_mask = ((torch.arange(T, device=device).view(1, T) > torch.arange(T, device=device).view(T, 1)) * FP_MIN).view(1, 1, T, T)
    attention_mask = causal_attention_mask[:,:,-1:,:]
    dst_attention_mask = causal_attention_mask[:,:,:,:1]
    
    compressed_mask = causal_topk_masking(
        estimated_probs, 
        k=K, 
        attention_mask=attention_mask, 
        dst_attention_mask=dst_attention_mask, 
        causal_attention_mask=causal_attention_mask
    )
    
    t = resize_from_m_to_t_csr(
        compressed_mask, 0, K,
        target_width=causal_attention_mask.shape[-1]
    )
    if t is not None:
        resized_mask_csr = t.to_dense().view(N, H, T, T)
    
    # print(resized_mask_csr)
    
    resized_mask = resize_from_m_to_t(
        compressed_mask, 0, 
        causal_attention_mask=causal_attention_mask,
        target_width=causal_attention_mask.shape[-1],
    )
    
    # print(resized_mask)
    # return
    
    # for i in range(H):
    #     print('-=-')
    #     print(resized_mask[0,i])
    #     print(resized_mask_csr[0,i])
    
    # return
    
    def bench_naive_convert():
        resized_mask = resize_from_m_to_t(
            compressed_mask, 0, 
            causal_attention_mask=causal_attention_mask,
            target_width=causal_attention_mask.shape[-1],
        )
        resized_mask = resized_mask.transpose(1, 2).reshape(N, T, H*T)
        return resized_mask.to_sparse_csr()
    
    def bench_csr_convert():
        return resize_from_m_to_t_csr(
            compressed_mask, 0, K,
            target_width=causal_attention_mask.shape[-1]
        )
    
    bench('csr_convert', bench_csr_convert, t_warmup=0.5, t_sample=3)
    bench('naive_convert', bench_naive_convert, t_warmup=0.5, t_sample=3)
    

if __name__ == '__main__':
    test_main()