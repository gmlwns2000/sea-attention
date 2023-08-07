import math
import torch
import torch.nn.functional as F

def flatten_masked_bmm_csr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor):
    assert mask.is_sparse_csr

    crow_indices = mask.crow_indices()
    col_indicies = mask.col_indices()
    out_values = mask.values()
    
    N, R_1 = crow_indices.shape
    N, Z = col_indicies.shape
    assert out_values.shape == (N, Z)
    
    for ir in range(R_1):
        for ic in range(crow_indices[R_1]):
            pass

def naive_flatten_masked_bmm_csr(a, b, mask):
    # a: N*H, T_DST, HID
    # b: N*H, T_SRC, HID
    # mask: N*H, T_DST, T_SRC
    
    score = torch.matmul(a, b.transpose(-1, -2))
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
    T = 2048
    T_DST = 2048
    T_M = 128
    K = 64
    HID = 64
    
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    device = 0
    
    # estimated_scores = torch.randn((N, H, T_DST, T_M), device=device)
    def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
    estimated_scores = F.interpolate(rand_perlin_2d((128, 128), (16, 16)).view(1, 1, 128, 128), (T_DST, T_M)).expand(N, H, T_DST, T_M).contiguous()
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
    key_layer = torch.randn((N, H, T_DST, HID), device=device)
    
    def bench_naive():
        return naive_flatten_masked_bmm_csr(
            query_layer, 
            key_layer, 
            torch.clamp_max(csr_mask.to_dense(), 1).view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
        )
    
    def bench_sparse():
        return flatten_masked_bmm_csr(query_layer, key_layer, csr_mask)
    
    score = bench_naive()
    score_sparse = bench_sparse()
    
    bench('naive_bmm', bench_naive, 0.5, 3)

if __name__ == '__main__':
    test_main()