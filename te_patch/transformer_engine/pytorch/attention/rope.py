import math
import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    """Lightweight Rotary Embedding compatible with NVIDIA ESM2 TE attention."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, max_seq_len: int, device=None, dtype=None):
        """Return (cos, sin) tensors shaped [1, 1, S, D]."""
        if dtype is None:
            dtype = torch.get_default_dtype()
        t = torch.arange(max_seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


__all__ = ["RotaryPositionEmbedding"]
