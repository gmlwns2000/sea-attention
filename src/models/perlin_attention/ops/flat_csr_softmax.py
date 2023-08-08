import torch

def naive_flatten_csr_softmax(scores: torch.Tensor):
    mask = scores == 0
    scores = scores.masked_fill(mask, torch.finfo(torch.float16).min * 0.5)
    probs = torch.softmax(scores, dim=-1).masked_fill_(mask, 0)
    return probs

def flatten_csr_softmax(scores: torch.Tensor):
    assert scores.is_sparse_csr
    crow_indices = scores.crow_indices()
    col_indices = scores.col_indices()
    in_values = scores.values()
    out_values = in_values.clone()
    N, R_1 = 

def test_main():
    from ....utils import seed
    from ....utils.bench import bench
    from .causal_resize_m_to_t import resize_from_m_to_t_csr
    from .causal_topk_masking import causal_topk_masking
    from .flat_csr_masked_bmm import flatten_csr_masked_bmm

    seed()
    
    N = 1
    H = 1
    T = 300
    T_DST = 300
    T_M = 4
    K = 2
    HID = 64
    
    # N = 1
    # H = 12
    # T = 2048
    # T_DST = 2048
    # T_M = 128
    # K = 64
    # HID = 64
    
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
    
    
    def bench_naive():
        with torch.no_grad():
            return naive_flatten_csr_softmax(
                csr_score.to_dense().view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
            )
    
    def bench_sparse():
        with torch.no_grad():
            return 
    
    probs = bench_naive()
    
    # print(score[0,0])
    # print(score_sparse[0,0,9,:])
    
    # max_error = (score - score_sparse).abs().max()
    # print(max_error)
    # if max_error > 1e-1:
    #     warnings.warn('max error exceed threshold')
    #     # for i in range(N):
    #     #     for j in range(H):
    #     #         for k in range(T_DST):
    #     #             for m in range(T):
    #     #                 err = (score[i,j,k,m] - score_sparse[i,j,k,m]).abs().item()
    #     #                 if err > 1e-1:
    #     #                     print(i,j,k,m,err,score[i,j,k,m],score_sparse[i,j,k,m])
    #     #                     return
    
    bench('sparse_softmax', bench_sparse, 0.5, 3)
    bench('naive_softmax', bench_naive, 0.5, 3)

if __name__ == '__main__':
    test_main()