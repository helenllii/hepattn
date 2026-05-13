import math
import torch
from torch import nn
import torch.nn.functional as F

# this is a trivial rewriting of https://github.com/lucidrains/linformer
# into a form that better fits the hepattn decoder setup

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class LinformerAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, q, k=None, v=None, attn_mask=None, **kwargs):
        #print("q.shape", q.shape)
        b, n, d, d_h, h, k_num = *q.shape, self.dim_head, self.heads, self.k

        kv_len = n if k is None else k.shape[1]
        if k is not None:
            assert v is not None, "v should not be None if k_input is not None"
        assert k.shape[1] == v.shape[1], f"{k.shape[1]} ?= {v.shape[1]}"
        assert kv_len <= self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(q)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        keys = self.to_k(k) if k is not None else self.to_k(q)
        values = self.to_v(v) if v is not None else self.to_v(q)

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # allow for variable sequence lengths (less than maximum sequence length) by slicing projections
        if kv_len < self.seq_len:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        # project keys and values along the sequence length dimension to k
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k_num, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        if attn_mask is not None:
            # Build mask in projected dots space.
            mask = torch.zeros_like(dots, dtype=torch.bool)

            original_mask = (attn_mask == 0)[:, None, ...]
            mask[..., :kv_len] = original_mask
            mask[..., kv_len:] = True

            # Rows where every position is masked.
            fully_masked = mask.all(dim=-1, keepdim=True)

            # Apply normal -inf masking.
            dots = dots.masked_fill(mask, float("-inf"))

            # Avoid softmax([-inf, -inf, ...]) = NaN.
            dots_for_softmax = torch.where(fully_masked, torch.zeros_like(dots), dots,)

            attn = dots_for_softmax.softmax(dim=-1)

            # Fully masked rows should attend to nothing.
            attn = torch.where(fully_masked, torch.zeros_like(attn),attn,)
        else:
            attn = dots.softmax(dim=-1)

        attn = self.dropout(attn)
        
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
