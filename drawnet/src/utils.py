"""
DrawNet Utilities
-----------------
Miscellaneous helpers: checkpoint I/O, seeding, logging, visualisation.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def seed_everything(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch: int, path: str, extra: dict = None):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved: {path}")


def load_checkpoint(model, optimizer, path: str, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    print(f"[Checkpoint] Loaded: {path}  (epoch {state.get('epoch', '?')})")
    return state.get("epoch", 0)


def plot_training_curves(train_losses, val_losses, save_path: str = None):
    """Plot and optionally save train/val loss curves."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DrawNet Training Curves")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def show_image_grid(images, titles=None, ncols: int = 5, figsize=(15, 6)):
    """Display a grid of PIL images or numpy arrays."""
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap="gray" if len(np.array(images[i]).shape) == 2 else None)
            if titles:
                ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
