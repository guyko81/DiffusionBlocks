"""Evaluation utilities: perplexity computation and inference-step sweep."""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_perplexity(model, dataloader: DataLoader, num_steps: int = None) -> float:
    """Compute token-level perplexity on a dataset."""
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        if hasattr(model, "diffusion_step"):
            logits = model.diffusion_step(input_ids, num_steps=num_steps)
        else:
            logits = model.model.forward_clean(input_ids)

        nll = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            targets.view(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += targets.numel()

    return math.exp(total_nll / total_tokens)


@torch.no_grad()
def sweep_inference_steps(model, dataloader: DataLoader, steps_list: list[int]) -> dict:
    """Sweep different numbers of inference steps, report perplexity for each."""
    results = {}
    for k in steps_list:
        ppl = compute_perplexity(model, dataloader, num_steps=k)
        results[k] = ppl
        print(f"  K={k}: PPL={ppl:.2f}")
    return results
