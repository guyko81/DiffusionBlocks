"""
Engram: O(1) hash-based memory module (DeepSeek, 2026).
N-gram hashing → embedding table lookup → learned gating → residual injection.
Table lives on CPU; gate trains end-to-end on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LMConfig
from .transformer import RMSNorm


HASH_PRIMES = [
    2654435761, 2246822519, 3266489917, 668265263,
    374761393, 2869860233, 2654435769, 805459861,
]


class EngramMemory(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.table_size = config.engram_table_size
        self.dim = config.hidden_size
        self.num_hashes = config.engram_num_hashes
        self.ngram_orders = config.engram_ngram_orders
        self.on_cpu = config.engram_on_cpu

        self.table = nn.Embedding(self.table_size, self.dim)
        nn.init.normal_(self.table.weight, std=0.02)

        self.register_buffer(
            "hash_primes",
            torch.tensor(HASH_PRIMES[: self.num_hashes], dtype=torch.long),
            persistent=False,
        )

        # Gating network (always on GPU)
        self.gate_norm_h = RMSNorm(self.dim)
        self.gate_norm_e = RMSNorm(self.dim)
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def _hash_ngrams(self, input_ids: torch.Tensor, order: int) -> torch.Tensor:
        """
        input_ids: [B, S] long tensor of token IDs.
        Returns: [B, S, num_hashes] indices into the table.
        """
        B, S = input_ids.shape
        padded = F.pad(input_ids, (order - 1, 0), value=0)
        indices = []
        for h in range(self.num_hashes):
            prime = self.hash_primes[h].item()
            acc = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)
            for k in range(order):
                acc = acc ^ (padded[:, k: k + S].long() * (prime >> k))
            indices.append(acc % self.table_size)
        return torch.stack(indices, dim=-1)

    def lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Hash → lookup → average over hashes and n-gram orders. Returns [B, S, D]."""
        results = []
        for order in self.ngram_orders:
            hash_idx = self._hash_ngrams(input_ids, order)  # [B, S, num_hashes]
            flat = hash_idx.reshape(-1)
            if self.on_cpu:
                flat = flat.cpu()
            embs = self.table(flat)  # [B*S*H, D]
            embs = embs.to(input_ids.device)
            embs = embs.view(*hash_idx.shape, self.dim)  # [B, S, H, D]
            results.append(embs.mean(dim=2))
        return torch.stack(results).mean(dim=0)  # [B, S, D]

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Gated residual injection.
        Returns: hidden_states + gate * engram_embedding
        """
        engram_emb = self.lookup(input_ids)
        h_norm = self.gate_norm_h(hidden_states)
        e_norm = self.gate_norm_e(engram_emb)
        gate_logit = (h_norm * e_norm).sum(dim=-1, keepdim=True) / (self.dim ** 0.5)
        gate = torch.sigmoid(gate_logit + self.gate_bias)
        return hidden_states + gate * engram_emb

    def move_table_to_cpu(self):
        """Call after model.to(device) to keep table on CPU with pinned memory."""
        if self.on_cpu:
            self.table = self.table.cpu()
            if torch.cuda.is_available():
                self.table.weight.data = self.table.weight.data.pin_memory()
