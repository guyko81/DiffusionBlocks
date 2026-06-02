"""CLI entry point for the DiffusionBlocks LM experiments."""

import argparse
from datetime import datetime, timezone, timedelta

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from lm.config import LMConfig
from lm.data_lm import WikiText103DataModule
from lm.model_lm import DBlockLM, BaselineLM


JST = timezone(timedelta(hours=9))


def main():
    parser = argparse.ArgumentParser(description="DiffusionBlocks LM")
    parser.add_argument("stage", choices=["train", "test"])
    parser.add_argument("--model_variant", choices=["baseline", "dblock", "dblock_engram"], default="dblock")

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--kv_lora_rank", type=int, default=128)
    parser.add_argument("--q_lora_rank", type=int, default=192)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)

    # Engram
    parser.add_argument("--engram_table_size", type=int, default=1_000_000)
    parser.add_argument("--engram_num_hashes", type=int, default=8)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", type=str, default="bf16-mixed")

    # Logging / checkpointing
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    # Inference
    parser.add_argument("--num_inference_steps", type=int, default=None)

    args = parser.parse_args()
    L.seed_everything(args.seed)

    # Build config
    config = LMConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        num_blocks=args.num_blocks,
        max_seq_len=args.seq_len,
        engram_enabled=(args.model_variant == "dblock_engram"),
        engram_table_size=args.engram_table_size,
        engram_num_hashes=args.engram_num_hashes,
    )

    # Data
    data = WikiText103DataModule(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
    )

    # Estimate total steps for scheduler
    # WikiText-103 ~103M tokens / (seq_len+1) = ~200K sequences, / batch_size = ~3125 steps/epoch
    est_steps_per_epoch = 200_000 // args.batch_size
    if args.model_variant == "baseline":
        max_epochs = args.num_epochs
    else:
        max_epochs = args.num_epochs * config.num_blocks
    total_steps = est_steps_per_epoch * max_epochs // args.accumulate_grad_batches

    # Model
    model_kwargs = dict(
        config=config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        scheduler_type=args.scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.model_variant == "baseline":
        model = BaselineLM(**model_kwargs)
    else:
        model = DBlockLM(**model_kwargs)

    # Experiment naming
    ts = datetime.now(JST).strftime("%Y-%m-%dT%H-%M-%S")
    exp_name = f"{ts}-lm-{args.model_variant}{args.postfix}"

    # Logger
    logger = WandbLogger(
        project="dblocks-lm",
        name=exp_name,
        offline=args.debug,
        save_dir="logs",
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"logs/{exp_name}",
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            save_last=True,
            every_n_epochs=args.save_every_n_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=100),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=0.25,
        enable_progress_bar=True,
        default_root_dir="logs",
    )

    if args.stage == "train":
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
    else:
        trainer.test(model, data, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
