"""Phase 2 — naive step-skipping accuracy/latency sweep for a DiffusionBlocks teacher.

Loads a trained ``ViTDBlockModel`` checkpoint *once* and evaluates CIFAR-100 test
accuracy at several inference-step counts by overriding the sampler schedule
(``model.sigmas``) per step count.

Why not just ``main.py test --num_inference_steps K``?
    ``model.sigmas`` is a *persistent* registered buffer, so it is baked into the
    checkpoint at the training-time step count (= ``num_blocks``). The stock test
    path rebuilds the model with the new ``--num_inference_steps K`` (buffer
    length K) and then loads the length-``num_blocks`` buffer from the ckpt,
    which mismatches shapes for any ``K != num_blocks``. Here we instead build
    the model at the default step count (so the load is clean) and then swap the
    schedule in Python before each evaluation.

Usage:
    python eval_steps.py --ckpt_path logs/<run>/last.ckpt --steps 3,2,1
"""

import argparse
import json
import time

import torch

from data import load_data
from dblock_modules import get_discrete_sigmas
from model import load_model


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument(
        "--steps",
        type=str,
        default="3,2,1",
        help="comma-separated inference-step counts to sweep",
    )
    p.add_argument("--data_name", type=str, default="cifar100")
    p.add_argument("--eval_batch_size", type=int, default=200)  # divides 10000 evenly
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--limit_batches",
        type=int,
        default=0,
        help="if >0, only evaluate this many batches (for quick plumbing checks)",
    )
    p.add_argument("--out", type=str, default=None, help="optional JSON output path")
    # dblock model args (must match the trained teacher)
    p.add_argument("--model_type", type=str, default="dblock")
    p.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="must match the trained checkpoint's ViT depth (default 12 for CIFAR)",
    )
    p.add_argument("--num_blocks", type=int, default=3)
    p.add_argument(
        "--weight_tied",
        action="store_true",
        help="must match the trained checkpoint (looped/recurrent-depth model)",
    )
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--cfg_scale", type=float, default=0.0)
    p.add_argument("--class_dropout_prob", type=float, default=0.0)
    args = p.parse_args()

    # fields the data/model constructors expect but we don't expose on the CLI
    args.batch_size = args.eval_batch_size
    args.add_rand_aug = False
    args.gradient_checkpointing = False
    args.num_inference_steps = None  # -> defaults to num_blocks, matching the ckpt buffer
    args.seed = 42
    return args


@torch.no_grad()
def evaluate(model, loader, device, limit_batches=0):
    """Return (top1_accuracy, n_examples, wall_seconds)."""
    model.eval()
    correct = total = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device).view(-1)
        logits = model.diffusion_step(x).view(-1, model.num_labels)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return correct / total, total, time.perf_counter() - t0


def main():
    args = build_args()
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    data = load_data(args)
    data.setup("test")
    args.image_size = data.image_size
    args.num_labels = data.num_labels

    model = load_model(args)
    model.configure_model()  # builds self.model; Lightning normally calls this

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model_missing = [k for k in missing if k.startswith("model.")]
    if model_missing:
        raise RuntimeError(f"Missing model weights in checkpoint: {model_missing[:5]} ...")
    if missing or unexpected:
        print(f"[load] non-fatal missing={missing} unexpected={unexpected}")
    model.to(device)

    loader = data.test_dataloader()
    print(f"Device: {device} | eval_batch_size={args.eval_batch_size} | steps={steps}")

    results = []
    for k in steps:
        # override the sampler schedule for this step count
        model.sigmas = get_discrete_sigmas(num_steps=k, dblock=True).to(device)
        acc, n, secs = evaluate(model, loader, device, limit_batches=args.limit_batches)
        ms_per_img = 1000.0 * secs / n
        results.append({"steps": k, "acc": acc, "n": n, "ms_per_img": ms_per_img})
        print(f"  steps={k:>2}  acc={acc*100:6.2f}%  ({n} ex)  {ms_per_img:6.3f} ms/img")

    # markdown table for the report
    print("\n| Inference steps | Top-1 acc | ms/img |")
    print("|---|---|---|")
    for r in results:
        print(f"| {r['steps']} | {r['acc']*100:.2f}% | {r['ms_per_img']:.3f} |")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt_path, "results": results}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
