import torch

def flatten_masked_bmm_csr(a, b, mask):
    assert mask.is_sparse

def naive_flatten_masked_bmm_csr(a, b, mask):
    # a: N*H, T_DST, HID
    # b: N*H, T_SRC, HID
    # mask: N*H, T_DST, T_SRC
    
    score = torch.bmm(a, b.transpose(-1, -2))
    score = score * mask
    return score

def test_main():
    from ....utils import seed
    from ....utils.bench import bench
    from .causal_resize_m_to_t import resize_from_m_to_t_csr
    from .causal_topk_masking import causal_topk_masking

    seed()
    
    # N = 1
    # H = 12
    # T = 8
    # T_DST = 3
    # T_M = 4
    # K = 4
    # HID = 64
    
    N = 2
    H = 12
    T = 1024
    T_DST = 1024
    T_M = 128
    K = 128
    HID = 64
    
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
    
    query_layer = torch.randn((N*H, T_DST, HID), device=device)
    key_layer = torch.randn((N*H, T_DST, HID), device=device)
    
    def bench_naive():
        return naive_flatten_masked_bmm_csr(query_layer, key_layer, csr_mask.to_dense().view(N, H, T_DST, T))
    
    score = bench_naive()
    
    bench('naive_bmm', naive_flatten_masked_bmm_csr, 0.5, 3)

if __name__ == '__main__':
    test_main()