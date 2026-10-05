#!/usr/bin/env python
"""Train one seq->expression arm (Step 4 probe, Step 5 escalation).

Examples:
  python train.py --arm decima --regime probe
  python train.py --arm borzoi --regime probe
  python train.py --arm decima --regime lora --epochs 15 --lr 4e-5

Checkpoints -> results/5_expression_models/models/<arm>_<regime>/.
Then evaluate with scripts/evaluate/evaluate_model.py.
"""

import argparse
import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from data import make_loaders
from model import SeqExpr

DATA_DIR = "/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome"
MODELS_DIR = f"{DATA_DIR}/results/5_expression_models/models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["decima", "borzoi"], required=True)
    ap.add_argument("--regime", choices=["probe", "lora", "full"], default="probe")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--accumulate", type=int, default=1)  # raise for 'full' on A30
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=8)     # regime=lora only
    ap.add_argument("--lora_conv", action="store_true")     # also adapt Conv1d (VERIFY peft support)
    args = ap.parse_args()

    tag = f"{args.arm}_{args.regime}"
    out = f"{MODELS_DIR}/{tag}"
    os.makedirs(out, exist_ok=True)
    print(f"=== training {tag} ===")

    loaders = make_loaders(batch_size=args.batch_size)
    model = SeqExpr(arm=args.arm, regime=args.regime, lr=args.lr,
                    lora_rank=args.lora_rank, lora_conv=args.lora_conv)

    ckpt = ModelCheckpoint(dirpath=out, monitor="val_loss", mode="min", save_top_k=1,
                           filename="best-{epoch:02d}-{val_loss:.4f}")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate,
        gradient_clip_val=1.0,
        # For --regime full on A30 24GB, also enable activation checkpointing in the
        # trunk (VERIFY: grelu/decima flag) to fit the 524 kb context.
        callbacks=[ckpt, EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")],
        log_every_n_steps=10,
        default_root_dir=out,
    )
    trainer.fit(model, loaders["train"], loaders["val"])
    print(f"best checkpoint: {ckpt.best_model_path}")
    with open(f"{out}/best_ckpt.txt", "w") as f:
        f.write(ckpt.best_model_path + "\n")


if __name__ == "__main__":
    main()
