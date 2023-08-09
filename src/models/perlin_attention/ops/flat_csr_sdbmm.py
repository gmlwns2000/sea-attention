import warnings
from cv2 import imreadmulti
import torch
import triton
import triton.language as tl

def __flatten_csr_sdbmm_py(
    crow_indices: torch.Tensor, 
    col_indices: torch.Tensor, 
    values: torch.Tensor, 
    other: torch.Tensor,
    output: torch.Tensor,
    N, R, Z, H, T_DST, T_SRC, HID,
):
    for n in range(N):
        for ir in range(R):
            crow_start = crow_indices[n, ir]
            crow_end = crow_indices[n, ir+1]
            
            cols = col_indices[n, crow_start:crow_end]
            col_values = values[n, crow_start:crow_end]
            head_idx = cols // T_SRC
            col_idx = cols % T_SRC
            
            for ih in range(H):
                head_mask = head_idx == ih
                other_head = other[n, ih]
                other_values = other_head[:,:]
                
                # too complicate in python

@triton.jit
def __flatten_csr_sdbmm_compute(
    CROW_INDICES,
    stride_crow_n, stride_crow_r,
    COL_INDICES,
    stride_col_n, stride_col_z,
    VALUES,
    stride_v_n, stride_v_z,
    OTHER,
    stride_other_n, stride_other_h, stride_other_t, stride_other_d,
    OUTPUT,
    stride_output_n, stride_output_h, stride_output_t, stride_output_d,
    TEMP_COUNT_HEAD,
    stride_tch_n, stride_tch_r, stride_tch_h,
    N, R, Z, H, T_DST, T_SRC, HID,
    MAX_ROW_Z: tl.constexpr, MAX_ROW_T: tl.constexpr, BLOCK_HID: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_R: tl.constexpr
):
    n = tl.program_id(0)
    pid_ir = tl.program_id(1)
    pid_hid = tl.program_id(2)
    
    for _ir in range(BLOCK_R):
        ir = pid_ir * BLOCK_R + _ir
        ir_mask = ir < R
        
        crow_start = tl.load(
            CROW_INDICES\
                + n*stride_crow_n\
                + ir*stride_crow_r,
            mask=ir_mask,
        )
        
        crow_end = tl.load(
            CROW_INDICES\
                + n*stride_crow_n\
                + (ir+1)*stride_crow_r,
            mask=ir_mask,
        )
        
        # compute head counts
        
        col_indices = tl.load(
            COL_INDICES\
                + n*stride_col_n\
                + (tl.arange(0, MAX_ROW_Z) + crow_start) * stride_col_z,
            mask=((tl.arange(0, MAX_ROW_Z) + crow_start) < crow_end) & ir_mask,
            other=T_SRC*BLOCK_H*2,
        )
        
        index_heads = col_indices // T_SRC
        
        count_heads = (index_heads[None,:] == tl.arange(0, BLOCK_H)[:, None]).to(tl.int32)
        count_heads_sum = tl.sum(count_heads, axis=1) # (BLOCK_H)
        count_heads_cumsum = tl.cumsum(count_heads_sum)
        tl.store(
            TEMP_COUNT_HEAD\
                + n*stride_tch_n\
                + ir*stride_tch_r\
                + tl.arange(0, BLOCK_H)*stride_tch_h,
            value=count_heads_cumsum,
            mask=(tl.arange(0, BLOCK_H) < H) & ir_mask
        )
        tl.static_assert(count_heads_cumsum.shape[0] == BLOCK_H)
        
        tl.debug_barrier()
        
        # perform per head column matmul
        
        for ih in range(H):
            ch_start = tl.load(
                TEMP_COUNT_HEAD\
                    + n*stride_tch_n\
                    + ir*stride_tch_r\
                    + (ih-1)*stride_tch_h,
                mask=((ih-1) >= 0) & ((ih-1) < H) & ir_mask,
                other=0
            )
            ch_end = tl.load(
                TEMP_COUNT_HEAD\
                    + n*stride_tch_n\
                    + ir*stride_tch_r\
                    + ih*stride_tch_h,
                mask=(ih < H) & ir_mask,
                other=-1
            )
            ch_end2 = tl.load(
                TEMP_COUNT_HEAD\
                    + n*stride_tch_n\
                    + ir*stride_tch_r\
                    + (ih+1)*stride_tch_h,
                mask=((ih+1) < H) & ir_mask,
                other=-1
            )
            ch_len = ch_end - ch_start
            
            per_head_col_indices_mask = tl.arange(0, MAX_ROW_T) < ch_len
            per_head_col_indices = tl.load(
                COL_INDICES\
                    + n*stride_col_n\
                    + (tl.arange(0, MAX_ROW_T) + ch_start + crow_start)*stride_col_z,
                mask=per_head_col_indices_mask & ir_mask,
                other=1
            ) % T_SRC
            
            row_values = tl.load(
                VALUES\
                    + n*stride_v_n\
                    + (tl.arange(0, MAX_ROW_T) + ch_start + crow_start)*stride_v_z,
                mask=per_head_col_indices_mask & ir_mask,
                other=0
            )
            
            hid_range = tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID
            hid_mask = hid_range < HID
            
            other_mask = ((per_head_col_indices_mask[:, None]) & (hid_mask[None, :]) & ir_mask)
            other_ptr = \
                per_head_col_indices[:,None]*stride_other_t\
                + hid_range[None,:]*stride_other_d
            # tl.debug_barrier()
            other = tl.load(
                OTHER\
                    + n*stride_other_n\
                    + ih*stride_other_h\
                    # + other_ptr,
                    + per_head_col_indices[:,None]*stride_other_t\
                    + hid_range[None,:]*stride_other_d,
                mask=other_mask,
                other=0
            ) # [MAX_ROW_T, BLOCK_HID]
            
            tl.static_assert(other.shape[0] == MAX_ROW_T)
            tl.static_assert(other.shape[1] == BLOCK_HID)
            
            other_mul = row_values[:, None] * other
            other_sum = tl.sum(other_mul, axis=0)
            
            tl.store(
                OUTPUT\
                    + n*stride_output_n\
                    + ih*stride_output_h\
                    + ir*stride_output_t\
                    + (tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID)*stride_output_d,
                    # + (tl.arange(0, MAX_ROW_T) + pid_hid * MAX_ROW_T)*stride_output_d,
                    # + (tl.arange(0, MAX_ROW_Z) + pid_hid * MAX_ROW_Z)*stride_output_d,
                # tl.sum(other * (tl.arange(0, MAX_ROW_T) == 1)[:, None], axis=0),
                # per_head_col_indices_mask,
                # stride_other_t,
                other_sum,
                # MAX_ROW_T,
                # stride_tch_n,
                # ih,
                # ch_end,
                # ir,
                mask=((tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID) < HID) & ir_mask,
                # mask=((tl.arange(0, MAX_ROW_T) + pid_hid * MAX_ROW_T) < HID) & ir_mask,
                # mask=((tl.arange(0, MAX_ROW_Z) + pid_hid * MAX_ROW_Z) < HID) & ir_mask,
            )
        

def flatten_csr_sdbmm(scores: torch.Tensor, value_layer: torch.Tensor, T_M: int, max_z_per_row:int=None):
    assert scores.is_sparse_csr
    crow_indices = scores.crow_indices()
    col_indices = scores.col_indices()
    values = scores.values()
    other = value_layer
    assert values.device == other.device
    N, R_1 = crow_indices.shape
    R = R_1 - 1
    N, Z = col_indices.shape
    
    _N, H, T_SRC, HID = other.shape
    assert N == _N
    _N, T_DST, HT_SRC = scores.shape
    assert N == _N
    assert HT_SRC == (H*T_SRC)
    output = torch.zeros((N, H, T_DST, HID), device=values.device)
    
    if max_z_per_row is None:
        max_z_per_row = (crow_indices[:,1:] - crow_indices[:,:-1]).max().item()
    
    # __flatten_csr_sdbmm_py(
    #     crow_indices, col_indices, values, other, output,
    #     N, R, Z, H, T_DST, T_SRC, HID
    # )
    
    # print(crow_indices[1])
    # print(col_indices[1, 50:], col_indices.shape)
    
    BLOCK_R = 8
    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_HID = triton.next_power_of_2(HID)
    MAX_ROW_Z = triton.next_power_of_2(max_z_per_row)
    num_warps = 1
    if BLOCK_H * BLOCK_HID < 1024:
        num_warps = 1
    elif BLOCK_H * BLOCK_HID < 2048:
        num_warps = 2
    elif BLOCK_H * BLOCK_HID < 4096:
        num_warps = 4
    elif BLOCK_H * BLOCK_HID  < 8192:
        num_warps = 8
    else:
        num_warps = 8
    # print(num_warps, BLOCK_H*BLOCK_HID * MAX_ROW_Z)
    MAX_ROW_T = triton.next_power_of_2(max(triton.cdiv(max_z_per_row, H)*2+1, triton.cdiv(T_SRC, T_M)))
    grid = (N, triton.cdiv(R, BLOCK_R), triton.cdiv(HID, BLOCK_HID))
    n_program = grid[0] * grid[1] * grid[2]
    
    # TODO this canbe reduced by reducing number of program in R dim
    temp_count_head = torch.zeros((N, R, H), dtype=torch.int32, device=values.device).fill_(42)
    # print(temp_count_head.shape)
    
    # print(n_program)
    # if n_program < 32:
    #     num_warps = 1
    # elif n_program < 64:
    #     num_warps = 2
    # else:
    #     num_warps = 2
    # print(grid, max_z_per_row)
    # print(grid)
    __flatten_csr_sdbmm_compute[grid](
        crow_indices,
        crow_indices.stride(0),crow_indices.stride(1),
        col_indices,
        col_indices.stride(0), col_indices.stride(1),
        values,
        values.stride(0), values.stride(1),
        other,
        other.stride(0), other.stride(1), other.stride(2), other.stride(3),
        output,
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        temp_count_head,
        temp_count_head.stride(0), temp_count_head.stride(1), temp_count_head.stride(2),
        N, R, Z, H, T_DST, T_SRC, HID,
        MAX_ROW_Z, MAX_ROW_T, BLOCK_HID, BLOCK_H, BLOCK_R,
        num_warps=num_warps,
        # num_stages=,
    )
    
    # print('tch', temp_count_head[1])
    del temp_count_head
    
    return output

def naive_flatten_csr_sdbmm(scores, values):
    return torch.matmul(scores, values)

def test_main():
    from ....utils import seed
    from ....utils.bench import bench
    from .causal_resize_m_to_t import resize_from_m_to_t_csr
    from .causal_topk_masking import causal_topk_masking
    from .flat_csr_masked_bmm import flatten_csr_masked_bmm
    from .flat_csr_softmax import flatten_csr_softmax
    from .flat_csr_to_dense import flatten_csr_to_dense

    seed()
    
    # N = 2
    # H = 6
    # T = 16
    # T_DST = 16
    # T_M = 3
    # K = 1
    # HID = 16
    
    N = 1
    H = 12
    T = 4096
    T_DST = 4096
    T_M = 128
    K = 64
    HID = 256
    
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    device = 0
    
    estimated_scores = torch.randn((N, H, T_DST, T_M), device=device)
    estimated_probs = torch.softmax(estimated_scores, dim=-1)
    causal_attention_mask = ((torch.arange(T, device=device).view(1, T) > torch.arange(T, device=device).view(T, 1)) * FP_MIN).view(1, 1, T, T)
    causal_attention_mask = causal_attention_mask[:, :, -T_DST:, :]
    attention_mask = causal_attention_mask[:,:,-1:,:]
    dst_attention_mask = causal_attention_mask[:,:,:,:1]
    
    compressed_mask = causal_topk_masking(
        estimated_probs, 
        k=K, 
        attention_mask=attention_mask, 
        dst_attention_mask=dst_attention_mask, 
        causal_attention_mask=causal_attention_mask
    )
    
    csr_mask = resize_from_m_to_t_csr(
        compressed_mask, 0, K,
        target_width=causal_attention_mask.shape[-1]
    )
    
    query_layer = torch.randn((N, H, T_DST, HID), device=device)
    key_layer = torch.randn((N, H, T, HID), device=device)
    csr_score = flatten_csr_masked_bmm(
        query_layer, 
        key_layer, 
        csr_mask, 
        None
    )
    csr_probs = flatten_csr_softmax(csr_score, H, T)
    # csr_probs_dense = csr_probs.to_dense().view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
    csr_probs_dense = flatten_csr_to_dense(csr_probs, T, H)
    
    value_layer = torch.randn((N, H, T, HID), device=device)
    
    def bench_naive():
        with torch.no_grad():
            return naive_flatten_csr_sdbmm(
                csr_probs_dense,
                value_layer
            )
    
    def bench_sparse():
        with torch.no_grad():
            return flatten_csr_sdbmm(csr_probs, value_layer, T_M)
    
    context = bench_naive()
    context_sparse = bench_sparse()
    idx_batch = 0
    idx_head = 0
    # print(value_layer[idx_batch, idx_head])
    # print(csr_probs_dense[idx_batch,idx_head])
    # print(context[idx_batch,idx_head])
    # print(context_sparse[idx_batch,idx_head])
    
    max_error = (context - context_sparse).abs().max()
    print(max_error, context.shape, context_sparse.shape)
    if max_error > 1e-1:
        warnings.warn('max error exceed threshold')
        for i in range(N):
            for j in range(H):
                for k in range(T_DST):
                    for m in range(HID):
                        err = (context[i,j,k,m] - context_sparse[i,j,k,m]).abs().item()
                        if err > 1e-1:
                            print(i,j,k,m,err,context[i,j,k,m],context_sparse[i,j,k,m])
                            return
    
    bench('sparse_sdbmm', bench_sparse, 0.5, 3, 'ms')
    bench('naive_sdbmm', bench_naive, 0.5, 3, 'ms')

if __name__ == '__main__':
    test_main()