import torch
import torch.nn as nn
import math
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))    
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
        res = x / rms * self.weight
        return res.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    def _SiLU(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def _GLU(self, x: torch.Tensor) -> torch.Tensor:
        return self._SiLU(self.w1(x)) * self.w3(x)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self._GLU(x))

class RoPE(nn.Module):
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        pos_index = torch.arange(max_seq_len, device=device)
        dim_index = torch.arange(0, d_k, 2, device=device)
        inv_freq = 1.0 / (theta ** (dim_index / d_k))
        freqs = einsum(pos_index, inv_freq, "seq_len , dim -> seq_len dim")
        
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_pos: torch.Tensor|None = None) -> torch.Tensor:
        if token_pos is None:
            seq_len = x.size(-2)
            token_pos = torch.arange(seq_len, device=x.device)
        cos = self.cos_cache[token_pos]  # (seq_len, d_k/2)
        sin = self.sin_cache[token_pos]  # (seq_len, d_k/2)
        x_odd = x[..., 1::2]  
        x_even = x[..., 0::2] 
        res1 = x_even * cos - x_odd * sin
        res2 = x_even * sin + x_odd * cos

        out = torch.zeros_like(x)
        out[..., 0::2] = res1
        out[..., 1::2] = res2
        return out
    
def softmax(x: torch.Tensor, dim: int)->torch.Tensor:
    x_exp = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V:torch.Tensor, 
    mask: torch.Tensor | None = None
    ) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = einsum(Q, K, "... seqq d_k, ... seqk d_k -> ... seqq seqk") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    output = einsum(attn_weights, V, "... seqq seqk, ... seqk d_v -> ... seqq d_v")
    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        _d_model = num_heads * self.d_k
        self.w_q = Linear(d_model, _d_model)
        self.w_k = Linear(d_model, _d_model)
        self.w_v = Linear(d_model, _d_model)
        self.w_o = Linear(_d_model, d_model)
        
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        token_pos: torch.Tensor | None = None
    )-> torch.Tensor:
        Q=self.w_q(x)
        K=self.w_k(x)
        V=self.w_v(x)
        
        Q = rearrange(Q, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)
        
        if self.rope is not None:
            Q = self.rope(Q, token_pos)
            K = self.rope(K, token_pos)
                
        if mask is None:
            seq_len = x.size(-2)
            mask = torch.ones(seq_len, seq_len, device=x.device).tril()
        
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = rearrange(attn_output, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        
        output = self.w_o(attn_output)
        return output
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        if theta is not None and max_seq_len is not None:
            self.attention = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        else:
            self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        attn_output = self.attention(self.ln1(x))
        x = x + attn_output
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output    
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        self.softmax = softmax
    
    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.linear(x)
        return logits