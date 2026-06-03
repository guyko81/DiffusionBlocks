"""
DiffusionBlocks causal LM with looped/weight-tied transformer + MLA + optional Engram.

Architecture follows the original DiffusionBlocks paper:
  - Clean context (input tokens) and noisy target (next-token embeddings)
    occupy SEPARATE positions in the sequence, mirroring CLS-vs-patches in ViT.
  - The model denoises the target positions while attending to clean context.
"""

import math
import random
import numpy as np
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import get_scheduler

from .config import LMConfig
from .transformer import CausalTransformer
from .engram import EngramMemory

import sys
sys.path.insert(0, ".")
from dblock_modules import get_block_sigmas, get_discrete_sigmas


class DBlockLM(L.LightningModule):
    """
    Looped DiffusionBlocks language model.
    Training: interleaved clean-context / noisy-target tokens, per-block denoiser.
    Inference: Euler ODE through K iterations.
    """

    def __init__(self, config: LMConfig, lr: float = 3e-4, weight_decay: float = 0.1,
                 warmup_steps: int = 1000, total_steps: int = 100000,
                 scheduler_type: str = "cosine", gradient_checkpointing: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type

        self.model = CausalTransformer(config, gradient_checkpointing=gradient_checkpointing)
        self.engram = EngramMemory(config) if config.engram_enabled else None

        self.block_sigmas = get_block_sigmas(config.num_blocks)
        sigmas = get_discrete_sigmas(config.num_blocks, dblock=True)
        self.register_buffer("sigmas", sigmas.float())

        self.sigma_data = config.sigma_data
        self.gamma = config.gamma

    def on_fit_start(self):
        if self.engram is not None:
            self.engram.move_table_to_cpu()

    def get_sigmas(self, n_samples: int, p_mean: float = -1.2, p_std: float = 1.2):
        block_idx = random.randint(0, self.config.num_blocks - 1)
        sigma_min_b = self.block_sigmas[block_idx]
        sigma_max_b = self.block_sigmas[block_idx + 1]

        if self.gamma > 0.0:
            log_min = np.log(sigma_min_b)
            log_max = np.log(sigma_max_b)
            log_range = log_max - log_min
            sigma_min_b = max(np.exp(log_min - self.gamma * log_range), self.block_sigmas[0])
            sigma_max_b = min(np.exp(log_max + self.gamma * log_range), self.block_sigmas[-1])

        cdf_min = norm.cdf((np.log(sigma_min_b) - p_mean) / p_std)
        cdf_max = norm.cdf((np.log(sigma_max_b) - p_mean) / p_std)
        u = np.random.uniform(cdf_min, cdf_max, n_samples)
        sigma = np.exp(p_mean + p_std * norm.ppf(u))
        return torch.from_numpy(sigma).float(), block_idx

    def get_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def shared_step(self, batch, step="train"):
        input_ids = batch["input_ids"]  # [B, S]
        targets = batch["targets"]      # [B, S]
        B, S = input_ids.shape

        # Target: L2-normalized embedding of next token
        with torch.no_grad():
            z = self.model.token_emb(targets)
            z = F.normalize(z, p=2, dim=-1)

        # Sample sigma from random block
        sigmas, block_idx = self.get_sigmas(B)
        sigmas = sigmas.to(z.device, z.dtype)

        # Corrupt target
        noise = torch.randn_like(z)
        zt = z + sigmas[:, None, None] * noise  # [B, S, D]

        # Karras preconditioner
        c_skip = self.sigma_data ** 2 / (sigmas ** 2 + self.sigma_data ** 2)
        c_out = sigmas * self.sigma_data / (sigmas ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1.0 / (sigmas ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = 0.25 * sigmas.log()

        # Engram
        engram_ctx = None
        if self.engram is not None and block_idx in self.config.engram_inject_iterations:
            engram_ctx = self.engram(input_ids, self.model.token_emb(input_ids))
            engram_ctx = engram_ctx - self.model.token_emb(input_ids)

        # Forward: interleaved context + noisy target (separate positions)
        model_out_logits = self.model(
            input_ids=input_ids,
            noisy_z=zt * c_in[:, None, None],
            sigma=c_noise,
            engram_ctx=engram_ctx,
        )

        # Weighted CE loss
        loss_flat = F.cross_entropy(
            model_out_logits.view(-1, self.config.vocab_size), targets.view(-1), reduction="none"
        )
        loss_per_sample = loss_flat.view(B, S).mean(dim=1)
        w = self.get_weights(sigmas)
        loss = (loss_per_sample * w).mean()

        self.log(f"{step}/loss", loss, prog_bar=True)
        self.log(f"{step}/loss_block_{block_idx}", loss)
        self.log(f"{step}/sigma_mean", sigmas.mean())
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    @torch.no_grad()
    def diffusion_step(self, input_ids: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """Full K-step Euler ODE inference. Returns logits [B, S, V]."""
        B, S = input_ids.shape
        D = self.config.hidden_size

        if num_steps is not None:
            sigmas = get_discrete_sigmas(num_steps, dblock=True).float().to(input_ids.device)
        else:
            sigmas = self.sigmas

        # Start from noise
        z = torch.randn(B, S, D, device=input_ids.device, dtype=self.dtype)
        z = z * (1.0 + sigmas[0] ** 2).sqrt()

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i].expand(B)
            next_sigma = sigmas[i + 1].expand(B)

            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
            c_in = 1.0 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
            c_noise = 0.25 * sigma.log()

            engram_ctx = None
            if self.engram is not None and i in self.config.engram_inject_iterations:
                engram_ctx = self.engram(input_ids, self.model.token_emb(input_ids))
                engram_ctx = engram_ctx - self.model.token_emb(input_ids)

            logits = self.model(
                input_ids=input_ids,
                noisy_z=z * c_in[:, None, None],
                sigma=c_noise,
                engram_ctx=engram_ctx,
            )

            # Denoised estimate via Karras preconditioner:
            # D(z) = c_skip * z + c_out * F(z*c_in)
            # But F outputs logits, so reconstruct embedding: probs @ emb_weight
            probs = F.softmax(logits, dim=-1)
            model_emb = probs @ self.model.token_emb.weight  # [B, S, D]
            denoised = c_skip[:, None, None] * z + c_out[:, None, None] * model_emb

            # Euler step
            d = (z - denoised) / sigma[:, None, None]
            dt = (next_sigma - sigma)[:, None, None]
            z = z + dt * d

        # Final denoise
        final_sigma = sigmas[-1].expand(B)
        c_in = 1.0 / (final_sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = 0.25 * final_sigma.log()

        engram_ctx = None
        if self.engram is not None and (len(sigmas) - 1) in self.config.engram_inject_iterations:
            engram_ctx = self.engram(input_ids, self.model.token_emb(input_ids))
            engram_ctx = engram_ctx - self.model.token_emb(input_ids)

        logits = self.model(
            input_ids=input_ids,
            noisy_z=z * c_in[:, None, None],
            sigma=c_noise,
            engram_ctx=engram_ctx,
        )
        return logits

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        engram_table_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "engram" in name and "table" in name:
                engram_table_params.append(param)
            elif "norm" in name or "bias" in name or "gate_bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))

        if engram_table_params:
            self.engram_optimizer = torch.optim.SparseAdam(
                engram_table_params, lr=self.lr * 0.1
            )

        scheduler = get_scheduler(
            self.scheduler_type, optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class BaselineLM(L.LightningModule):
    """Standard autoregressive LM (no DiffusionBlocks, no looping) for ablation."""

    def __init__(self, config: LMConfig, lr: float = 3e-4, weight_decay: float = 0.1,
                 warmup_steps: int = 1000, total_steps: int = 100000,
                 scheduler_type: str = "cosine", gradient_checkpointing: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type

        self.model = CausalTransformer(config, gradient_checkpointing=gradient_checkpointing)

    def shared_step(self, batch, step="train"):
        input_ids = batch["input_ids"]
        targets = batch["targets"]
        logits = self.model.forward_clean(input_ids)
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        self.log(f"{step}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr, betas=(0.9, 0.95),
        )
        scheduler = get_scheduler(
            self.scheduler_type, optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
