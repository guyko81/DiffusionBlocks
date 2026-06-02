"""
Decoder-only transformer with:
  - Multi-head Latent Attention (MLA, DeepSeek-V2 style)
  - Dense SwiGLU FFN
  - AdaLN sigma conditioning (DiffusionBlocks)
  - Decoupled RoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import LMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x: [B, S, H, D_rope] — apply rotary to pairs."""
    seq_len = x.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class MultiHeadLatentAttention(nn.Module):
    """
    MLA: compresses KV into a shared low-rank latent before caching.
    Decoupled RoPE applied on a separate projection path.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim  # d_rope + d_nope
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim

        # Query path: down-project then up-project
        self.q_down = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_nope_up = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        self.q_rope_up = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)

        # KV path: down-project to shared latent, then up-project
        self.kv_down = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.k_nope_up = nn.Linear(config.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        self.v_up = nn.Linear(config.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)

        # Decoupled RoPE path for keys (separate from latent)
        self.k_rope_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_rope_head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

        self.scale = (self.qk_rope_head_dim + self.qk_nope_head_dim) ** -0.5

    def forward(self, h: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        B, S, _ = h.shape

        # Query: compress then expand
        c_q = self.q_norm(self.q_down(h))
        q_nope = self.q_nope_up(c_q).view(B, S, self.num_heads, self.qk_nope_head_dim)
        q_rope = self.q_rope_up(c_q).view(B, S, self.num_heads, self.qk_rope_head_dim)
        q_rope = apply_rope(q_rope, rope_cos, rope_sin)
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, S, H, head_dim]

        # KV: compress to shared latent then expand
        c_kv = self.kv_norm(self.kv_down(h))
        k_nope = self.k_nope_up(c_kv).view(B, S, self.num_heads, self.qk_nope_head_dim)
        v = self.v_up(c_kv).view(B, S, self.num_heads, self.v_head_dim)

        # Decoupled RoPE for keys
        k_rope = self.k_rope_proj(h).view(B, S, self.num_heads, self.qk_rope_head_dim)
        k_rope = apply_rope(k_rope, rope_cos, rope_sin)
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, S, H, head_dim]

        # Scaled dot-product attention with causal mask
        q = q.transpose(1, 2)  # [B, H, S, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # [B, H, S, v_head_dim]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale
        )  # [B, H, S, v_head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TimestepEmbedder(nn.Module):
    """Maps scalar sigma → conditioning vector via Fourier features + MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class DeepSeekAdaLNBlock(nn.Module):
    """Single decoder block: MLA + SwiGLU + AdaLN sigma conditioning."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config)
        self.ffn = SwiGLUFFN(config)
        # AdaLN: 6 modulation scalars (shift/scale/gate for attn and ffn)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size),
        )
        # Zero-init the AdaLN output so the block starts as identity
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln(cond).chunk(6, dim=-1)

        # Attention sub-block with AdaLN
        h = self.attn_norm(x)
        h = h * (1.0 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        h = self.attn(h, rope_cos, rope_sin)
        x = x + gate_a.unsqueeze(1) * h

        # FFN sub-block with AdaLN
        h = self.ffn_norm(x)
        h = h * (1.0 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)
        h = self.ffn(h)
        x = x + gate_f.unsqueeze(1) * h

        return x


class CausalTransformer(nn.Module):
    """
    Full decoder stack: token embedding + U shared AdaLN layers + output head.
    Designed for DiffusionBlocks looped inference (same layers run K times).
    """

    def __init__(self, config: LMConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.time_emb = TimestepEmbedder(config.hidden_size)
        self.layers = nn.ModuleList([
            DeepSeekAdaLNBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.output_head.weight = self.token_emb.weight

        # Precompute RoPE frequencies
        rope_cos, rope_sin = precompute_rope_freqs(
            config.qk_rope_head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, noisy_z: torch.Tensor,
                sigma: torch.Tensor, engram_ctx: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids: [B, S] context tokens
        noisy_z:   [B, S, D] noisy target embeddings (scaled by c_in)
        sigma:     [B] c_noise values (0.25 * log(sigma))
        engram_ctx: [B, S, D] optional engram residual (pre-gated)
        Returns: logits [B, S, V]
        """
        x = self.token_emb(input_ids) + noisy_z
        if engram_ctx is not None:
            x = x + engram_ctx

        cond = self.time_emb(sigma)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cond, self.rope_cos, self.rope_sin, use_reentrant=False)
            else:
                x = layer(x, cond, self.rope_cos, self.rope_sin)

        x = self.final_norm(x)
        return self.output_head(x)

    def forward_clean(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard forward without denoising (for baseline training)."""
        x = self.token_emb(input_ids)
        cond = torch.zeros(x.shape[0], self.config.hidden_size, device=x.device)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cond, self.rope_cos, self.rope_sin, use_reentrant=False)
            else:
                x = layer(x, cond, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        return self.output_head(x)
