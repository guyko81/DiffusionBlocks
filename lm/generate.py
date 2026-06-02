"""Text generation utilities for DBlockLM."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    num_steps: int = None,
):
    """
    Autoregressive generation using DiffusionBlocks inference.
    prompt_ids: [1, prompt_len] tensor of token IDs.
    """
    model.eval()
    generated = prompt_ids.clone()
    max_seq = model.config.max_seq_len

    for _ in range(max_new_tokens):
        ctx = generated[:, -max_seq:]
        logits = model.diffusion_step(ctx, num_steps=num_steps)
        next_logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            top_vals, top_idx = next_logits.topk(top_k)
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits.scatter_(1, top_idx, top_vals)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = next_logits.sort(descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            mask = cumprobs - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            next_logits = sorted_logits.gather(1, sorted_indices.argsort(1))

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_token], dim=1)

    return generated
