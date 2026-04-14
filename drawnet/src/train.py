"""
DrawNet Training Script
-----------------------
Trains the DrawNet multi-task model on QuickDraw + TU-Berlin data.

Prerequisites (run once before training):
    python src/build_annotations.py --show_stats
    python src/cache_dataset.py
    python src/split_dataset.py --show_stats

Usage
-----
    cd drawnet/
    python src/train.py
    python src/train.py --config configs/config.yaml --device cuda
    python src/train.py --resume outputs/checkpoints/epoch_020.pt

Backbone unfreezing schedule
-----------------------------
    Epochs  1  - UNFREEZE_PHASE1  : backbone frozen, heads only
    Epochs  UNFREEZE_PHASE1+1 - UNFREEZE_PHASE2 : unfreeze layer4
    Epochs  UNFREEZE_PHASE2+1 - end              : unfreeze full backbone
"""

import argparse
import pathlib
import time
import yaml
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_XM = None   # torch_xla handle — set in main() if --device tpu

from model    import DrawNet, DrawNetLoss
from dataset  import CachedDataset, build_dataloaders, DEVIATION_CLASSES
from utils    import seed_everything, save_checkpoint, load_checkpoint, plot_training_curves
from evaluate import evaluate_intent, evaluate_deviation

UNFREEZE_PHASE1 = 10    # after this epoch -> unfreeze layer4
UNFREEZE_PHASE2 = 20    # after this epoch -> unfreeze full backbone
LR_BACKBONE     = 1e-5  # lower LR once backbone is unfrozen


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = intent_loss_sum = dev_loss_sum = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, intent_labels, deviation_labels) in enumerate(loader):
        images           = images.to(device)
        intent_labels    = intent_labels.to(device)
        deviation_labels = deviation_labels.to(device)

        optimizer.zero_grad()
        intent_logits, deviation_logits = model(images)
        loss, l_intent, l_dev = criterion(
            intent_logits, deviation_logits, intent_labels, deviation_labels
        )
        loss.backward()

        if _XM:
            _XM.optimizer_step(optimizer)
            _XM.mark_step()
        else:
            optimizer.step()

        total_loss      += loss.item()
        intent_loss_sum += l_intent.item()
        dev_loss_sum    += l_dev.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx+1}/{n_batches}  "
                  f"loss={loss.item():.4f}  "
                  f"intent={l_intent.item():.4f}  "
                  f"dev={l_dev.item():.4f}")

    n = n_batches
    return total_loss / n, intent_loss_sum / n, dev_loss_sum / n


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = intent_loss_sum = dev_loss_sum = 0.0

    all_intent_logits = []
    all_intent_labels = []
    all_dev_logits    = []
    all_dev_labels    = []

    for images, intent_labels, deviation_labels in loader:
        images           = images.to(device)
        intent_labels    = intent_labels.to(device)
        deviation_labels = deviation_labels.to(device)

        intent_logits, deviation_logits = model(images)
        loss, l_intent, l_dev = criterion(
            intent_logits, deviation_logits, intent_labels, deviation_labels
        )

        total_loss      += loss.item()
        intent_loss_sum += l_intent.item()
        dev_loss_sum    += l_dev.item()

        all_intent_logits.append(intent_logits.cpu())
        all_intent_labels.append(intent_labels.cpu())
        all_dev_logits.append(deviation_logits.cpu())
        all_dev_labels.append(deviation_labels.cpu())

    n      = len(loader)
    losses = (total_loss / n, intent_loss_sum / n, dev_loss_sum / n)

    intent_metrics = evaluate_intent(
        torch.cat(all_intent_logits), torch.cat(all_intent_labels)
    )
    dev_metrics = evaluate_deviation(
        torch.cat(all_dev_logits), torch.cat(all_dev_labels)
    )

    metrics = {f"intent_{k}": v for k, v in intent_metrics.items()}
    metrics.update({f"dev_{k}": v for k, v in dev_metrics.items()
                    if not isinstance(v, list)})

    for name, auc in zip(DEVIATION_CLASSES, dev_metrics.get("per_class_auroc", [])):
        metrics[f"auroc_{name}"] = auc

    return losses, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(42)

    global _XM
    if args.device == "tpu":
        import torch_xla.core.xla_model as xm
        _XM    = xm
        device = xm.xla_device()
        print(f"Device: TPU ({device})")
    else:
        device = torch.device(
            args.device if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_cfg    = cfg["data"]
    cache_dir   = pathlib.Path(data_cfg["cache_dir"])
    train_csv   = str(cache_dir / "train.csv")
    test_csv    = str(cache_dir / "test.csv")

    for p in [train_csv, test_csv]:
        if not pathlib.Path(p).exists():
            print(f"Missing: {p}")
            print("Run build_annotations.py -> cache_dataset.py -> split_dataset.py first.")
            return

    print("\nBuilding dataloaders ...")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_csv   = train_csv,
        test_csv    = test_csv,
        train_frac  = data_cfg["train_split"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["training"]["num_workers"],
        image_size  = data_cfg["image_size"],
    )

    # Compute pos_weights from training split only
    train_ds    = train_loader.dataset.dataset   # unwrap Subset -> CachedDataset
    pos_weights = train_ds.compute_pos_weights().to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model ...")
    model = DrawNet(
        num_intent_classes    = cfg["model"]["num_intent_classes"],
        num_deviation_classes = cfg["model"]["num_deviation_classes"],
        pretrained            = cfg["model"]["pretrained"],
        freeze_backbone       = cfg["model"]["freeze_layers"],
    ).to(device)

    criterion = DrawNetLoss(
        pos_weight = pos_weights,
        lambda1    = cfg["training"]["lambda1"],
        lambda2    = cfg["training"]["lambda2"],
    )

    optimizer = AdamW(
        model.trainable_params(),
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        peek         = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_epoch = peek.get("epoch", 0)
        del peek

        if resume_epoch >= UNFREEZE_PHASE1 + 1:
            model.unfreeze_phase(1)
            optimizer.add_param_group({
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": LR_BACKBONE,
            })
        if resume_epoch >= UNFREEZE_PHASE2 + 1:
            model.unfreeze_phase(2)
            tracked = {id(p) for g in optimizer.param_groups for p in g["params"]}
            new_params = [p for p in model.backbone.parameters()
                          if p.requires_grad and id(p) not in tracked]
            if new_params:
                optimizer.add_param_group({"params": new_params, "lr": LR_BACKBONE})

        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed from epoch {start_epoch}")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir  = pathlib.Path("outputs/logs")
    ckpt_dir = pathlib.Path("outputs/checkpoints")
    res_dir  = pathlib.Path("outputs/results")
    for d in [log_dir, ckpt_dir, res_dir]:
        d.mkdir(parents=True, exist_ok=True)

    history       = []
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    total_epochs = cfg["training"]["epochs"]
    print(f"\nStarting training for {total_epochs} epochs ...\n")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()

        # Progressive backbone unfreezing
        if epoch == UNFREEZE_PHASE1 + 1:
            model.unfreeze_phase(1)
            optimizer.add_param_group({
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": LR_BACKBONE,
            })
        elif epoch == UNFREEZE_PHASE2 + 1:
            model.unfreeze_phase(2)
            tracked = {id(p) for g in optimizer.param_groups for p in g["params"]}
            new_params = [p for p in model.backbone.parameters()
                          if p.requires_grad and id(p) not in tracked]
            if new_params:
                optimizer.add_param_group({"params": new_params, "lr": LR_BACKBONE})

        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_losses, val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        row = {
            "epoch":        epoch,
            "train_loss":   train_losses[0],
            "train_intent": train_losses[1],
            "train_dev":    train_losses[2],
            "val_loss":     val_losses[0],
            "val_intent":   val_losses[1],
            "val_dev":      val_losses[2],
            "lr":           optimizer.param_groups[0]["lr"],
            "time_s":       elapsed,
            **val_metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{total_epochs}  "
            f"train={train_losses[0]:.4f}  val={val_losses[0]:.4f}  "
            f"intent_top1={val_metrics.get('intent_top1_acc', 0):.3f}  "
            f"dev_f1={val_metrics.get('dev_macro_f1', 0):.3f}  "
            f"({elapsed:.0f}s)"
        )

        if val_losses[0] < best_val_loss:
            best_val_loss = val_losses[0]
            save_checkpoint(model, optimizer, epoch,
                            str(ckpt_dir / "best.pt"),
                            extra={"val_loss": best_val_loss})

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch,
                            str(ckpt_dir / f"epoch_{epoch:03d}.pt"))

        pd.DataFrame(history).to_csv(log_dir / "train_log.csv", index=False)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation ...")
    load_checkpoint(model, None, str(ckpt_dir / "best.pt"), device)
    _, test_metrics = validate(model, test_loader, criterion, device)

    print("\n=== TEST SET RESULTS ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    pd.DataFrame([test_metrics]).to_csv(res_dir / "test_metrics.csv", index=False)

    plot_training_curves(
        [r["train_loss"] for r in history],
        [r["val_loss"]   for r in history],
        save_path=str(res_dir / "training_curves.png"),
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
