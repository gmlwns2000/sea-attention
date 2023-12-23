import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from performer_pytorch import FastAttention
import numpy as np

def imshow_pixels(ps, psmask, need_expand):
    return

    # ps = ps * psmask.long()
    # ret = torch.empty((N, T_DST, T_SRC), device=ps.device).fill_(0)
    # ret.scatter_(dim=-1, index=ps, value=1)
    # ret = ret.cpu().numpy()
    # def max_pool(img, factor: int):
    #     """ Perform max pooling with a (factor x factor) kernel"""
    #     ds_img = np.full((img.shape[0] // factor, img.shape[1] // factor), -float('inf'), dtype=img.dtype)
    #     np.maximum.at(ds_img, (np.arange(img.shape[0])[:, None] // factor, np.arange(img.shape[1]) // factor), img)
    #     return ds_img
    # ret = max_pool(ret[0], 1)
    # plt.clf()
    # plt.imshow(ret)
    # plt.savefig("hello.png", dpi=320)
    # input(">>>")

@torch.compile
def masked_bmm_masked_indices(a: Tensor, b: Tensor, xs: Tensor, xs_mask: Tensor, value: float):
    """
    a: [N, A, B]
    b: [N, C, B]
    
    xs: [N, A, Z] < C
    xs_mask: [N, A, Z] \in {0, 1}
    value: float
    
    - return
    ys: [N, A, Z] \in { (R := a[i, :] \dot b[:, j]) if xs_mask == 1 else value }
    """
    
    N, A, B = a.shape
    _N, C, _B = b.shape
    assert B == _B
    assert N == _N
    _N, _A, Z = xs.shape
    assert N == _N
    assert A == _A
    
    xs = torch.clamp(xs, 0, C - 1)
    
    # # [N, A, 1, B]
    # xs_a = a.view(N, A, 1, B)
    # # [N, A, Z, B]
    # xs_b = b.view(N, 1, C, B).expand(N, A, C, B)
    # xs_b_idx = xs.view(N, A, Z, 1).expand(N, A, Z, B)
    # xs_b = xs_b.gather(dim=-2, index=xs_b_idx)
    
    # ys = (xs_a * xs_b).sum(-1)
    # ys = torch.where(xs_mask > 0.5, ys, value)
    # return ys
    
    SPARQ = 32
    _, a_idx = torch.topk(a.abs(), k=SPARQ, dim=-1)
    a_mask = torch.zeros_like(a)
    a_mask.scatter_(dim=-1, index=a_idx, value=1)
    a = a_mask * a
    
    ts = torch.bmm(a, b.transpose(-1, -2))
    ys = ts.gather(dim=-1, index=xs, sparse_grad=True)
    ys = torch.where(xs_mask > 0.5, ys, value)
    return ys

@torch.compile
def forward_mask(self_w: int, self_k: int, q: Tensor, k: Tensor):
    device = q.device
    dtype = q.dtype
    N, T_SRC, HID = k.shape
    N, T_DST, HID = q.shape
    
    start_t_src = T_SRC - T_DST + 1
    end_t_src = start_t_src + T_DST
    tsrcs = torch.arange(start_t_src, end_t_src, device=device)[None, :, None].expand(N, T_DST, 1)
    
    w = self_w
    ws = torch.empty((T_DST,), device=device, dtype=torch.float)[None, :, None].fill_(self_w).expand(N, T_DST, 1)
    # w cannot be larger than t_src
    
    assert self_w <= self_k
    OVERSAMPLE = 1.0
    SCALE_UP = 2
    max_pixels = int(round(self_k * OVERSAMPLE))
    pixels = torch.arange(0, max_pixels, device=device)[None, None, :].expand(N, T_DST, max_pixels).contiguous()
    pixels_mask = torch.empty_like(pixels).fill_(0).float()
    pixels_mask[:, :, :self_w] = 1.0
    # pixel_counts = torch.empty((T_DST,), device=device, dtype=torch.long).fill_(self.w)[None, :, None].expand(N, T_DST, 1).contiguous()
    
    need_expand = ws < tsrcs
    while True:
        # imshow_pixels(pixels, pixels_mask, need_expand)
        
        # TODO: topk masked pixels
        t_tsrcs = torch.masked_select(tsrcs, need_expand).view(N, -1, 1)
        tws = torch.masked_select(ws, need_expand).view(N, -1, 1)
        # print('a', t_tsrcs.shape, pixels.shape)
        tpixels = torch.masked_select(pixels, need_expand)
        tpixels = tpixels.view(N, -1, pixels.shape[-1])
        tpixels_mask = torch.masked_select(pixels_mask, need_expand).view(N, -1, pixels.shape[-1])
        tq = torch.masked_select(q, need_expand).view(N, -1, q.shape[-1])
        
        # print(need_expand.shape, tpixels.shape, t_tsrcs.shape, tws.shape)
        txs = torch.round(tpixels * (t_tsrcs / tws)).long()
        scores = masked_bmm_masked_indices(tq, k, txs, tpixels_mask, -32000.0)
        # k = clamp(round(tws / t_tsrcs * self.k)) * (1 if tww == t_tsrcs else 1.5), 1, tws)
        tks = torch.clamp(
            torch.round((tws / t_tsrcs * self_k) * torch.where(tws == t_tsrcs, 1, OVERSAMPLE)), 
            torch.tensor(1, device=device), 
            torch.clamp_max(tws - 1, scores.shape[-1] - 1)
        ).long()
        # print(torch.round((tws / t_tsrcs * self.k) * torch.where(tws == t_tsrcs, 1, OVERSAMPLE))[0, :, 0], self.k, tks.max().item(), scores.shape)
        values, indices = torch.topk(scores, k=tks.max().item(), dim=-1, sorted=True, largest=True)
        new_pixels = pixels.gather(dim=-1, index=indices)
        new_pixels_mask = (torch.arange(0, indices.shape[-1], device=device)[None, None, :] < tks) * 1.0
        # print(new_pixels_mask[0, -1, :])
        
        new_pixels = F.pad(new_pixels, (0, pixels.shape[-1] - new_pixels.shape[-1]), value=0.0)
        new_pixels_mask = F.pad(new_pixels_mask, (0, pixels.shape[-1] - new_pixels_mask.shape[-1]), value=0.0)
        # print(pixels.shape, need_expand.shape, new_pixels_mask.shape)
        pixels = torch.masked_scatter(pixels, need_expand, new_pixels)
        # print(pixels_mask.shape, need_expand.shape, new_pixels_mask.shape)
        pixels_mask = torch.masked_scatter(pixels_mask, need_expand, new_pixels_mask)
        # print(pixels_mask[0, -1, :])
        
        # need_expand = ws < tsrcs
        ws_new = torch.min(tsrcs, ws * SCALE_UP)
        
        # TODO: resize from ws_new to ws
        N, A, Z = pixels.shape
        scale = ws_new / ws
        ps_start = pixels * scale
        ps_end = (pixels + 1) * scale
        ps_start = torch.round(ps_start).long()
        ps_end = torch.round(ps_end).long()
        ps = ps_start[:, :, :, None].expand(N, A, Z, 2)
        ps_counts = (ps_end - ps_start)[:, :, :, None]
        reps = torch.arange(0, 2, device=device)[None, None, None, :]
        reps = reps * pixels_mask[:, :, :, None]
        ps = ps + torch.clamp_max(reps, torch.clamp_min(ps_counts - 1, 0)).long()
        ps = ps.view(N * A, Z * 2).contiguous()
        ps, _ = torch.sort(ps, dim=-1, descending=False)
        
        _, indices = torch.unique_consecutive(ps, return_inverse=True)
        indices -= indices.min(dim=1, keepdim=True)[0]
        result = -torch.ones_like(ps)
        ps = result.scatter_(1, indices, ps)
        ps = ps.view(N, A, -1)
        
        pixels_mask = (ps >= 0) * 1.0
        pixels = ps * pixels_mask.long()
        max_z = int(pixels_mask.sum(-1).max().item())
        pixels = pixels[..., :max_z].contiguous()
        pixels_mask = pixels_mask[..., :max_z].contiguous()
        # print(pixels[0, -1], pixels.shape)
        # input()
        
        ws = ws_new

        # manage break
        if w == T_SRC:
            break
        w = min(T_SRC, w * SCALE_UP)
    
    # imshow_pixels(pixels, pixels_mask, need_expand)
    
    pixels = pixels * pixels_mask.long()
    ret = torch.empty((N, T_DST, T_SRC), device=device).fill_(-32000.0)
    ret.scatter_(dim=-1, index=pixels, value=0)
    return ret

class TreeAttention(nn.Module):
    def __init__(self, causal: bool, k: int, w: int):
        super().__init__()
        
        self.causal = causal
        self.k = k
        self.w = w
        assert causal
        
        self.performer = FastAttention(
            dim_heads=64,
            nb_features=64,
            causal=self.causal,
            generalized_attention=self.causal,
        )
    
    def forward_single_quary(
        self, 
        q: Tensor, 
        kcum: Tensor,
        k: Tensor, 
        v: Tensor
    ):
        assert q.ndim == 3
        assert k.ndim == 3
        assert v.ndim == 3
        assert q.shape[1] == 1
        assert k.shape[1] == v.shape[1]
        
        if k.shape[1] < self.k:
            score = torch.bmm(q, k.transpose(-1, -2))
            probs = torch.softmax(score, dim=-1)
            context = torch.bmm(probs, v)
            return context, probs
        
        N, T_SRC, HID = k.shape
        N, T_DST, HID = q.shape
        
        w = 16
        assert self.w > w
        assert w <= T_SRC
        
        scores_mask = torch.ones((N, 1, w), dtype=torch.float32, device=q.device)
        
        while True:
            KEY_APPROX_METHOD = 'skip'
            if KEY_APPROX_METHOD == 'avg':
                # key avg
                idx = torch.round(torch.arange(0, w+1, dtype=torch.float32, device=q.device) * (T_SRC / (w+1))).long()
                start_idx = idx[:-1]
                end_idx = idx[1:]
                pixel_counts = end_idx - start_idx
                
                start_idx = start_idx[None, :, None].expand(N, w, HID)
                end_idx = end_idx[None, :, None].expand(N, w, HID)
                pixel_counts = pixel_counts[None, :, None].expand(N, w, HID)
                
                tk = kcum.gather(dim=1, index=end_idx) - kcum.gather(dim=1, index=start_idx)
                tk = tk / (pixel_counts + 1e-12)
            elif KEY_APPROX_METHOD == 'skip':
                # key skipping
                idx = torch.round(torch.arange(0, w, dtype=torch.float32, device=q.device) * (T_SRC / (w))).long()
                idx = idx[None, :, None].expand(N, w, HID)
                tk = k.gather(dim=1, index=idx)
            else:
                raise Exception()
            
            # sparq
            SPARQ = 32
            values, indices = torch.topk(q.abs(), dim=-1, k=SPARQ)
            tq = q.gather(dim=-1, index=indices)
            tk = tk.gather(dim=-1, index=indices.expand(N, tk.shape[1], -1))
            
            # perform attention
            scores = torch.softmax(
                torch.bmm(tq, tk.transpose(-1, -2))\
                    .masked_fill_(scores_mask < 0.5, torch.finfo(q.dtype).min),
                dim=-1
            )
            
            # sink attention
            SINKING = 2
            if SINKING > 0:
                scores[:, :, :SINKING] = 2.0
            values, indices = torch.topk(scores, k=int(round(min(w, max(1, round(w / T_SRC * self.k)) * (1 if w == T_SRC else 1.5)))), dim=-1)
            scores_mask.fill_(0)
            scores_mask.scatter_(dim=-1, index=indices, value=1)
            
            w = min(T_SRC, w * 2)
            scores_mask = F.interpolate(scores_mask, size=(w,), mode='nearest')
            if w == T_SRC:
                break
        
        scores = torch.bmm(q, k.transpose(-1, -2))
        scores = scores.masked_fill_(scores_mask < 0.5, torch.finfo(q.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        context = torch.bmm(probs, v)
        
        # # calc alpha with sparq
        # SPARQ = 16
        # values, indices = torch.topk(q.abs(), dim=-1, k=SPARQ)
        # tq = q.gather(dim=-1, index=indices)
        # tk = k.gather(dim=-1, index=indices.expand(N, k.shape[1], -1))
        # tprobs = torch.softmax(torch.bmm(tq, tk.transpose(-1, -2)), -1)
        # alpha = (scores_mask * tprobs).sum(-1, keepdim=True)
        # avg_context = v.sum(dim=-2, keepdim=True)
        
        # context = context * alpha + avg_context * (1 - alpha)
        
        return context, probs
    
    def forward_single(self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor):
        assert q.ndim == 4
        assert k.ndim == 4
        assert v.ndim == 4
        assert k.shape[-2] == v.shape[-2]
        
        N, H, T_SRC, HID = k.shape
        N, H, T_DST, HID = q.shape
        
        kcum = k.cumsum(dim=-2).view(N*H, T_SRC, HID)
        
        a_rows = []
        c_rows = []
        for t in range(T_DST):
            if (t % 1000) == 0:
                print(t)
            assert T_SRC-T_DST+t+1 > 0
            t_src = T_SRC-T_DST+t+1
            k_row = k[..., :t_src, :]
            v_row = v[..., :t_src, :]
            q_row = q[..., t:t+1, :]
            k_row = k_row.view(N*H, t_src, HID)
            v_row = v_row.view(N*H, t_src, HID)
            q_row = q_row.view(N*H, 1, HID)
            
            c_row, a_row = self.forward_single_quary(
                q=q_row,
                kcum=kcum, 
                k=k_row,
                v=v_row
            )
            c_rows.append(c_row)
            
            # a_row = F.pad(a_row, pad=(0, T_SRC - a_row.shape[-1]), value=0)
            # a_rows.append(a_row)
        
        # a_rows = torch.concat(a_rows, dim=-2)
        # score = torch.matmul(q, k.transpose(-1, -2)).view(N, H, T_DST, T_SRC) + attention_mask
        # probs = torch.softmax(score, dim=-1).view(N*H, T_DST, T_SRC)
        # torch.save({'a': a_rows, 's': probs}, 'dump.pth')
        
        context = torch.concat(c_rows, dim=-2)
        context = context.view(N, H, T_DST, HID)
        
        # performer
        with torch.autocast("cuda", torch.float32):
            pcontext = self.performer(
                q.to(torch.float32), 
                k.to(torch.float32), 
                v.to(torch.float32),
            )
        context = pcontext * 0.1 + context * 0.9
        
        return context
    
    def forward_batch(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        __q = q
        __k = k
        __v = v
        __mask = mask
        N, H, T_SRC, HID = k.shape
        N, H, T_DST, HID = q.shape
        
        contexts = []
        t_dense = max(0, self.k - T_SRC + T_DST)
        if t_dense > 0:
            q, q_dense = q[..., t_dense:, :], q[..., :t_dense, :]
            mask, mask_dense = mask[..., t_dense:, :], mask[..., :t_dense, :]
            scores = torch.matmul(q_dense, k.transpose(-1, -2))
            scores = scores + mask_dense
            probs = torch.softmax(scores, -1)
            context = torch.matmul(probs, v)
            contexts.append(context)
        
        t_sparse = T_DST - t_dense
        if t_sparse > 0:
            mask_sparse = forward_mask(self.w, self.k, q.view(N*H, t_sparse, HID), k.view(N*H, T_SRC, HID))
            mask_sparse = mask_sparse.view(N, H, t_sparse, T_SRC)
            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores + mask_sparse
            probs = torch.softmax(scores, -1)
            context = torch.matmul(probs, v)
            contexts.append(context)
        
        contexts = torch.concat(contexts, dim=-2)
        
        # print(mask_dense.shape, mask_sparse.shape)
        # mask = torch.concat([
        #     mask_dense.expand(mask_sparse.shape[0], mask_sparse.shape[1], -1, -1), 
        #     mask_sparse
        # ], dim=-2)
        # mask = (mask > -1) * 1.0
        
        # scores = torch.matmul(__q, __k.transpose(-1, -2))
        # scores = scores + mask
        # approx = torch.softmax(scores, dim=-1)
        
        # scores = torch.matmul(__q, __k.transpose(-1, -2))
        # scores = scores + __mask
        # probs = torch.softmax(scores, dim=-1)
        
        # torch.save({'a': approx, 's': probs}, 'dump.pth')
        
        return contexts
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor):
        # return self.forward_single(q, k, v, attention_mask)
        return self.forward_batch(q, k, v, attention_mask)