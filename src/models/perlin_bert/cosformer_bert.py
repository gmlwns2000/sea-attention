import warnings
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

class CosformerAttention(nn.Module):
    def __init__(
            self, 
            embed_dim):
        super().__init__()
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.act_fun = None

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        
    def forward(
        self,
        q : torch.Tensor,
        k : torch.Tensor,
        v : torch.Tensor,
        eps : torch.Optional[float] = 1e-6,
        act_fun = "relu",
        has_outproj = True,
        causal = False,
        attn_mask : torch.Optional[torch.Tensor] =None, # for causal attention
    ):
        self.act_fun = self.get_act_fun(act_fun)
        warnings.warn(f"act_fun {act_fun}")
        warnings.warn(f"has_outproj {has_outproj}")
        warnings.warn(f"causal {causal}")

        
        N, H, T, HID = q.shape
        q = q.permute(2, 0, 1, 3).contiguous().view(T, N, H*HID) # [T, N, H, HID]->[T, N, H*HID] # NOTE if the shape is same, can we think the effect of each projection is same?
        k = k.permute(2, 0, 1, 3).contiguous().view(T, N, H*HID)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        if act_fun == 'elu':
            q = q+1 # NOTE JIN the result shows 0.002 higher than expected
            k = k+1
        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, N * H, HID).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, N * H, HID).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, N * H, HID).transpose(0, 1)
        
        # cos transform
        tgt_len = T # TODO tgt_len ?
        src_len = T
        eps = 1e-6
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        if causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, N, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, N, -1) # T, N, H*HID
        # L, N, E
        if has_outproj:
            attn_output = self.out_proj(attn_output)

        # [N, H, T, HID]
        context_layer = attn_output.view(T, N, H, HID).permute(1, 2, 0, 3) # N, H, T, HID

        attention_probs = torch.zeros((N, H, T, T), device=q.device, dtype=q.dtype)

        return context_layer, attention_probs