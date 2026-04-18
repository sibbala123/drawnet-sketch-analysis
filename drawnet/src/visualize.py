"""
DrawNet Visualization
---------------------
Generates two types of diagnostic outputs:

1. Grad-CAM heatmaps — shows where the model looks when predicting intent
   and detecting each deviation. One output grid per deviation class + one
   for intent.

2. Confusion matrix + per-class accuracy — full 30-class intent confusion
   matrix, per-class accuracy bar chart, and top-10 confused pairs.

Usage
-----
    cd drawnet/
    python src/visualize.py --gradcam
    python src/visualize.py --confusion
    python src/visualize.py --gradcam --confusion   # both at once
    python src/visualize.py --gradcam --n_samples 8 --checkpoint output/checkpoints/best.pt
"""

import argparse
import pathlib
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model   import DrawNet
from dataset import get_eval_transforms, DEVIATION_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_NAMES = [
    "face", "house", "tree", "cat", "car",
    "bird", "dog", "fish", "flower", "bicycle",
    "airplane", "butterfly", "elephant", "horse", "rabbit",
    "bear_(animal)", "mushroom", "cup", "umbrella", "shoe",
    "guitar", "snake", "spider", "apple", "sun",
    "bridge", "castle", "palm_tree", "eye", "chair",
]


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Computes Grad-CAM heatmaps for a target layer.
    Hooks into the target layer to capture activations and gradients.
    """

    def __init__(self, model: DrawNet, target_layer: torch.nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, image_tensor: torch.Tensor, target_score: torch.Tensor) -> np.ndarray:
        """
        Parameters
        ----------
        image_tensor : (1, 3, H, W)
        target_score : scalar tensor with grad_fn (intent logit or deviation logit)

        Returns
        -------
        cam : (H, W) numpy array normalized to [0, 1]
        """
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # Global average pool gradients over spatial dims -> (C,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)  # (H, W)
        cam     = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        cam_np = cam.cpu().numpy()
        H, W   = image_tensor.shape[-2], image_tensor.shape[-1]
        cam_resized = np.array(
            Image.fromarray(cam_np).resize((W, H), Image.BILINEAR)
        )
        return cam_resized

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend Grad-CAM heatmap over the original image."""
    cmap       = plt.cm.jet(cam)[..., :3]          # (H, W, 3) RGB, values [0,1]
    img_norm   = image_np.astype(np.float32) / 255.
    blended    = alpha * cmap + (1 - alpha) * img_norm
    return np.clip(blended, 0, 1)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and return HWC uint8 numpy array."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def run_gradcam(
    model:      DrawNet,
    df:         pd.DataFrame,
    device:     torch.device,
    out_dir:    pathlib.Path,
    n_samples:  int = 8,
    dev_classes: list = None,
):
    """
    Generate Grad-CAM grids:
      - One grid per deviation class (showing images where that deviation is present)
      - One grid for intent (showing correct predictions with high confidence)
    """
    if dev_classes is None:
        dev_classes = DEVIATION_CLASSES

    transform = get_eval_transforms(224)

    # Target layer: last conv block of ResNet-50 layer4
    target_layer = list(model.backbone.children())[7][-1].conv3
    gradcam      = GradCAM(model, target_layer)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Deviation Grad-CAM ────────────────────────────────────────────────────
    for dev_idx, dev_name in enumerate(dev_classes):
        if dev_name not in df.columns:
            print(f"[Grad-CAM] Column '{dev_name}' not in CSV — skipping.")
            continue

        pos_df = df[df[dev_name] == 1].sample(
            min(n_samples, (df[dev_name] == 1).sum()), random_state=42
        )
        if len(pos_df) == 0:
            print(f"[Grad-CAM] No positive samples for {dev_name} — skipping.")
            continue

        n_cols = min(n_samples, len(pos_df))
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6))
        fig.suptitle(f"Grad-CAM — Deviation: {dev_name}", fontsize=14, fontweight="bold")

        for col_i, (_, row) in enumerate(pos_df.iterrows()):
            if col_i >= n_cols:
                break

            img_pil    = Image.open(row["filepath"]).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            img_np     = denormalize(img_tensor.squeeze(0))

            img_tensor.requires_grad_(True)
            intent_logits, dev_logits = model(img_tensor)
            target_score = dev_logits[0, dev_idx]

            cam     = gradcam.compute(img_tensor, target_score)
            overlay = overlay_heatmap(img_np, cam)

            axes[0, col_i].imshow(img_np)
            axes[0, col_i].axis("off")
            intent_pred = CATEGORY_NAMES[intent_logits.argmax(1).item()] \
                          if intent_logits.shape[1] == len(CATEGORY_NAMES) else str(intent_logits.argmax(1).item())
            axes[0, col_i].set_title(intent_pred, fontsize=8)

            axes[1, col_i].imshow(overlay)
            axes[1, col_i].axis("off")
            score = torch.sigmoid(dev_logits[0, dev_idx]).item()
            axes[1, col_i].set_title(f"score: {score:.2f}", fontsize=8)

        axes[0, 0].set_ylabel("Original", fontsize=9)
        axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)

        plt.tight_layout()
        save_path = out_dir / f"gradcam_{dev_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Grad-CAM] Saved: {save_path}")

    # ── Intent Grad-CAM ───────────────────────────────────────────────────────
    # One sample per category, then take n_samples from those
    per_class = []
    for lbl in sorted(df["intent_label"].unique()):
        group = df[df["intent_label"] == lbl]
        per_class.append(group.sample(1, random_state=42))
    sample_df = (pd.concat(per_class, ignore_index=True)
                   .sample(min(n_samples, len(per_class)), random_state=42)
                   .reset_index(drop=True))

    n_cols = len(sample_df)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Grad-CAM — Intent Recognition", fontsize=14, fontweight="bold")

    for col_i, (_, row) in enumerate(sample_df.iterrows()):
        img_pil    = Image.open(row["filepath"]).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        img_np     = denormalize(img_tensor.squeeze(0))

        img_tensor.requires_grad_(True)
        intent_logits, _ = model(img_tensor)
        pred_class   = intent_logits.argmax(1).item()
        target_score = intent_logits[0, pred_class]

        cam     = gradcam.compute(img_tensor, target_score)
        overlay = overlay_heatmap(img_np, cam)

        true_label = int(row["intent_label"])
        pred_name  = CATEGORY_NAMES[pred_class] if pred_class < len(CATEGORY_NAMES) else str(pred_class)
        true_name  = CATEGORY_NAMES[true_label] if true_label < len(CATEGORY_NAMES) else str(true_label)
        correct    = pred_class == true_label

        axes[0, col_i].imshow(img_np)
        axes[0, col_i].axis("off")
        color = "green" if correct else "red"
        axes[0, col_i].set_title(true_name, fontsize=8)

        axes[1, col_i].imshow(overlay)
        axes[1, col_i].axis("off")
        conf = F.softmax(intent_logits, dim=1)[0, pred_class].item()
        axes[1, col_i].set_title(f"{pred_name}\n{conf:.0%}", fontsize=8, color=color)

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)

    plt.tight_layout()
    save_path = out_dir / "gradcam_intent.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Grad-CAM] Saved: {save_path}")

    gradcam.remove_hooks()


# ---------------------------------------------------------------------------
# Confusion Matrix + Per-Class Accuracy
# ---------------------------------------------------------------------------

def run_confusion(
    model:       DrawNet,
    df:          pd.DataFrame,
    device:      torch.device,
    out_dir:     pathlib.Path,
    batch_size:  int = 64,
):
    """
    Run inference on the full df, then produce:
      1. 30x30 intent confusion matrix (normalized by row)
      2. Per-class accuracy bar chart
      3. Top-10 confused pairs table (printed + saved as text)
    """
    transform    = get_eval_transforms(224)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_preds  = []
    all_labels = []

    print(f"[Confusion] Running inference on {len(df)} samples ...")
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_rows = df.iloc[start:start + batch_size]
            imgs, lbls = [], []
            for _, row in batch_rows.iterrows():
                try:
                    img = Image.open(row["filepath"]).convert("RGB")
                    imgs.append(transform(img))
                    lbls.append(int(row["intent_label"]))
                except Exception:
                    continue

            if not imgs:
                continue

            batch = torch.stack(imgs).to(device)
            logits, _ = model(batch)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls)

            if (start // batch_size) % 10 == 0:
                print(f"  {start}/{len(df)}")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n_classes  = len(CATEGORY_NAMES)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        if true < n_classes and pred < n_classes:
            cm[true, pred] += 1

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(CATEGORY_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CATEGORY_NAMES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Intent Classification — Normalized Confusion Matrix", fontsize=14, fontweight="bold")

    # Annotate cells with values ≥ 0.10 to keep it readable
    for i in range(n_classes):
        for j in range(n_classes):
            if cm_norm[i, j] >= 0.10:
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Confusion] Saved: {cm_path}")

    # ── Per-class accuracy bar chart ──────────────────────────────────────────
    per_class_acc = cm_norm.diagonal()
    sorted_idx    = np.argsort(per_class_acc)
    colors        = ["#d62728" if a < 0.80 else "#2ca02c" for a in per_class_acc[sorted_idx]]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(
        [CATEGORY_NAMES[i] for i in sorted_idx],
        per_class_acc[sorted_idx] * 100,
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.axvline(per_class_acc.mean() * 100, color="black", linestyle="--",
               linewidth=1.5, label=f"Mean: {per_class_acc.mean()*100:.1f}%")
    ax.set_xlabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Intent Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 105)

    for bar, acc in zip(bars, per_class_acc[sorted_idx]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc*100:.1f}%", va="center", fontsize=8)

    red_patch   = mpatches.Patch(color="#d62728", label="< 80%")
    green_patch = mpatches.Patch(color="#2ca02c", label=">= 80%")
    ax.legend(handles=[red_patch, green_patch, plt.Line2D([0], [0], color="black",
              linestyle="--", label=f"Mean: {per_class_acc.mean()*100:.1f}%")], fontsize=9)

    plt.tight_layout()
    bar_path = out_dir / "per_class_accuracy.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Confusion] Saved: {bar_path}")

    # ── Top-10 confused pairs ──────────────────────────────────────────────────
    confusions = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], CATEGORY_NAMES[i], CATEGORY_NAMES[j]))

    confusions.sort(reverse=True)
    print("\n[Confusion] Top-10 most confused pairs:")
    print(f"  {'True':<20} {'Predicted':<20} {'Count':>6}  {'Rate':>6}")
    print(f"  {'-'*20} {'-'*20} {'------':>6}  {'------':>6}")
    for count, true_cat, pred_cat in confusions[:10]:
        true_idx = CATEGORY_NAMES.index(true_cat)
        rate     = cm_norm[true_idx, CATEGORY_NAMES.index(pred_cat)]
        print(f"  {true_cat:<20} {pred_cat:<20} {count:>6}  {rate*100:>5.1f}%")

    # Save summary
    summary_lines = [
        "Per-class accuracy summary",
        "=" * 40,
        f"Mean accuracy: {per_class_acc.mean()*100:.1f}%",
        f"Lowest: {CATEGORY_NAMES[sorted_idx[0]]} — {per_class_acc[sorted_idx[0]]*100:.1f}%",
        f"Highest: {CATEGORY_NAMES[sorted_idx[-1]]} — {per_class_acc[sorted_idx[-1]]*100:.1f}%",
        "",
        "Top-10 confused pairs:",
    ]
    for count, true_cat, pred_cat in confusions[:10]:
        true_idx = CATEGORY_NAMES.index(true_cat)
        rate = cm_norm[true_idx, CATEGORY_NAMES.index(pred_cat)]
        summary_lines.append(f"  {true_cat} -> {pred_cat}  ({count} samples, {rate*100:.1f}%)")

    summary_path = out_dir / "confusion_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"[Confusion] Saved: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradcam",    action="store_true", help="Generate Grad-CAM heatmaps")
    parser.add_argument("--confusion",  action="store_true", help="Generate confusion matrix")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (auto-detects best.pt)")
    parser.add_argument("--data_csv",   default=None,
                        help="CSV with filepath + intent_label + deviation columns. "
                             "Uses test.csv if available, else index.csv.")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--n_samples",  type=int, default=8,
                        help="Number of samples per Grad-CAM grid")
    parser.add_argument("--out_dir",    default=None,
                        help="Output directory (default: output/results/)")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    if not args.gradcam and not args.confusion:
        parser.print_help()
        print("\nSpecify --gradcam and/or --confusion")
        sys.exit(0)

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Locate checkpoint ─────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = pathlib.Path(args.checkpoint)
    else:
        candidates = [
            pathlib.Path("output/checkpoints/best.pt"),
            pathlib.Path("outputs/checkpoints/best.pt"),
        ]
        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            print("No checkpoint found. Pass --checkpoint <path>")
            sys.exit(1)
    print(f"Checkpoint: {ckpt_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DrawNet(
        num_intent_classes    = cfg["model"]["num_intent_classes"],
        num_deviation_classes = cfg["model"]["num_deviation_classes"],
        pretrained            = False,
        freeze_backbone       = False,
    )
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    print(f"Model loaded (epoch {state.get('epoch', '?')})")

    # ── Locate data CSV ───────────────────────────────────────────────────────
    if args.data_csv:
        csv_path = pathlib.Path(args.data_csv)
    else:
        candidates = [
            pathlib.Path(cfg["data"]["cache_dir"]) / "test.csv",
            pathlib.Path("data/augmented/test.csv"),
            pathlib.Path(cfg["data"]["cache_dir"]) / "index.csv",
            pathlib.Path("data/augmented/index.csv"),
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            print("No data CSV found. Pass --data_csv <path>")
            sys.exit(1)
    print(f"Data CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Use deviation columns from config, falling back to whatever's in the CSV
    dev_cols = cfg.get("deviation_classes", DEVIATION_CLASSES)
    dev_cols = [c for c in dev_cols if c in df.columns]
    if not dev_cols:
        print(f"Warning: no deviation columns from config found in CSV. "
              f"CSV columns: {list(df.columns)}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = pathlib.Path(args.out_dir) if args.out_dir \
              else pathlib.Path("output/results")

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.gradcam:
        print("\n=== Grad-CAM ===")
        run_gradcam(model, df, device, out_dir / "gradcam",
                    n_samples=args.n_samples, dev_classes=dev_cols)

    if args.confusion:
        print("\n=== Confusion Matrix ===")
        # For confusion matrix use a representative subset if CSV is large
        sample_df = df if len(df) <= 15000 else df.groupby("intent_label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), 500), random_state=42)
        )
        run_confusion(model, sample_df, device, out_dir)


if __name__ == "__main__":
    main()
