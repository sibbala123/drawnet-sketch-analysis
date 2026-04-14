"""
DrawNet Dataset Classes
-----------------------
CachedDataset       : reads pre-cached PNGs + index.csv (produced by cache_dataset.py)
TUBerlinDataset     : loads TU-Berlin JPEG images directly from folder structure
NumpyBitmapDataset  : loads QuickDraw .npy bitmap files
build_dataloaders   : splits train.csv into train/val DataLoaders
                      and wraps test.csv in a test DataLoader

Data sources
------------
QuickDraw (labels 0-9)
    data/raw/quickdraw/{category}.npy   shape (N, 784) uint8

TU-Berlin (labels 10-29)
    data/raw/tuberlin/{category}/*.jpg

Both sources are pre-processed by:
    build_annotations.py  ->  data/processed/master_annotations.csv
    cache_dataset.py      ->  data/augmented/index.csv + PNGs
    split_dataset.py      ->  data/augmented/train.csv + test.csv
"""

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVIATION_CLASSES = [
    "rotation",
    "closure_failure",
    "spatial_disorganization",
    "size_distortion",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def numpy_bitmap_to_image(bitmap: np.ndarray, image_size: int = 224) -> Image.Image:
    """Convert a flat 784-element QuickDraw bitmap (28x28) to RGB PIL image."""
    arr = bitmap.reshape(28, 28).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize(
        (image_size, image_size), Image.LANCZOS
    )
    return Image.merge("RGB", [img, img, img])


# ---------------------------------------------------------------------------
# NumpyBitmapDataset  — QuickDraw .npy files (used by build_annotations.py)
# ---------------------------------------------------------------------------

class NumpyBitmapDataset(Dataset):
    """
    Loads QuickDraw drawings from .npy bitmap files.

    Returns (PIL image, intent_label_idx) pairs.
    Used by build_annotations.py and cache_dataset.py — not directly by train.py.

    Parameters
    ----------
    data_dir : str
        Directory containing .npy files.
    categories : List[str]
        Category names (must match filenames without extension).
    label_offset : int
        Added to the category index to get the global intent label.
        QuickDraw starts at 0, so offset=0.
    samples_per_class : int
        Max samples to load per category.
    image_size : int
        Output spatial resolution.
    """

    def __init__(
        self,
        data_dir: str,
        categories: List[str],
        label_offset: int = 0,
        samples_per_class: int = 5000,
        image_size: int = 224,
    ):
        self.image_size  = image_size
        self.label_offset = label_offset
        self.label_to_idx: Dict[str, int] = {
            c: i + label_offset for i, c in enumerate(categories)
        }
        self.samples: List[Tuple[np.ndarray, int]] = []
        self._load(Path(data_dir), categories, samples_per_class)

    def _load(self, data_dir: Path, categories: List[str], n: int):
        for cat in categories:
            path = data_dir / f"{cat}.npy"
            if not path.exists():
                print(f"[NumpyBitmapDataset] WARNING: {path} not found — skipping.")
                continue
            arr = np.load(path)[:n]
            label = self.label_to_idx[cat]
            for row in arr:
                self.samples.append((row, label))
        print(f"[NumpyBitmapDataset] Loaded {len(self.samples)} samples "
              f"from {len(categories)} categories.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        bitmap, label = self.samples[idx]
        return numpy_bitmap_to_image(bitmap, self.image_size), label


# ---------------------------------------------------------------------------
# TUBerlinDataset  — JPEG images from folder structure
# ---------------------------------------------------------------------------

class TUBerlinDataset(Dataset):
    """
    Loads TU-Berlin hand-drawn sketch images from a folder-per-category layout.

    Expected layout::

        data/raw/tuberlin/
            airplane/
                sketch_001.jpg
                ...
            butterfly/
                ...

    Returns (PIL image, intent_label_idx) pairs.
    Used by build_annotations.py and cache_dataset.py — not directly by train.py.

    Parameters
    ----------
    data_dir : str
        Root directory containing one subfolder per category.
    categories : List[str]
        Category folder names to load.
    label_offset : int
        Added to category index to get global intent label.
        TU-Berlin starts at 10 (after 10 QuickDraw classes), so offset=10.
    image_size : int
        Output spatial resolution for PIL resize.
    """

    def __init__(
        self,
        data_dir: str,
        categories: List[str],
        label_offset: int = 10,
        image_size: int = 224,
    ):
        self.image_size  = image_size
        self.label_offset = label_offset
        self.label_to_idx: Dict[str, int] = {
            c: i + label_offset for i, c in enumerate(categories)
        }
        self.samples: List[Tuple[Path, int]] = []
        self._load(Path(data_dir), categories)

    def _load(self, data_dir: Path, categories: List[str]):
        for cat in categories:
            cat_dir = data_dir / cat
            if not cat_dir.exists():
                print(f"[TUBerlinDataset] WARNING: {cat_dir} not found — skipping.")
                continue
            label = self.label_to_idx[cat]
            found = [p for p in cat_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            for p in found:
                self.samples.append((p, label))
        print(f"[TUBerlinDataset] Loaded {len(self.samples)} samples "
              f"from {len(categories)} categories.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        return img, label


# ---------------------------------------------------------------------------
# CachedDataset  — reads pre-cached PNGs + CSV (used by train.py / evaluate.py)
# ---------------------------------------------------------------------------

class CachedDataset(Dataset):
    """
    Reads pre-augmented images from disk (generated by cache_dataset.py).

    CSV columns::

        filepath, intent_label, rotation, closure_failure,
        spatial_disorganization, size_distortion

    Parameters
    ----------
    csv_path : str
        Path to train.csv or test.csv produced by split_dataset.py.
    transform : optional callable
        Applied to each PIL image at load time.
    """

    def __init__(self, csv_path: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform

        for col in DEVIATION_CLASSES:
            if col not in self.df.columns:
                self.df[col] = 0

        print(f"[CachedDataset] Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        dev_label = torch.tensor(
            [float(row[c]) for c in DEVIATION_CLASSES], dtype=torch.float32
        )
        return img, int(row["intent_label"]), dev_label

    def compute_pos_weights(self) -> torch.Tensor:
        """
        Per-class positive weights for BCEWithLogitsLoss.
        pos_weight[i] = (N - N_pos[i]) / N_pos[i], clipped to [1, 10].
        """
        total = len(self.df)
        weights = []
        print("pos_weights per deviation class:")
        for col in DEVIATION_CLASSES:
            n_pos = max(float(self.df[col].sum()), 1.0)
            w     = float(np.clip((total - n_pos) / n_pos, 1.0, 10.0))
            weights.append(w)
            print(f"  {col:30s}: {w:.2f}")
        return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_csv:   str,
    test_csv:    str,
    train_frac:  float = 0.80,
    batch_size:  int   = 32,
    num_workers: int   = 0,
    image_size:  int   = 224,
    seed:        int   = 42,
    tuberlin_label_start: int = 10,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from pre-split CSVs.

    train.csv is split internally into train (80%) and val (20%).
    test.csv is used as-is for final evaluation.

    A WeightedRandomSampler is applied to the training split to up-sample
    TU-Berlin categories (fewer images per class than QuickDraw).

    Parameters
    ----------
    train_csv : str
        Path to data/augmented/train.csv
    test_csv : str
        Path to data/augmented/test.csv
    train_frac : float
        Fraction of train.csv used for training; remainder is validation.
    tuberlin_label_start : int
        Intent label index where TU-Berlin categories begin (default 10).
        Used to assign higher sampling weight to TU-Berlin samples.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_transform = get_train_transforms(image_size)
    eval_transform  = get_eval_transforms(image_size)

    # ── Train / val split from train.csv ─────────────────────────────────────
    full_train_ds = CachedDataset(train_csv, transform=train_transform)
    total   = len(full_train_ds)
    n_train = int(total * train_frac)

    generator = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(total, generator=generator).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    from torch.utils.data import Subset
    train_ds = Subset(full_train_ds, train_idx)
    val_ds   = Subset(full_train_ds, val_idx)

    # ── Weighted sampler: up-sample TU-Berlin in training ─────────────────────
    # TU-Berlin categories (labels >= tuberlin_label_start) have fewer images,
    # so we give them higher weight so they're seen proportionally during training.
    df = full_train_ds.df
    label_counts = df["intent_label"].value_counts().to_dict()
    max_count    = max(label_counts.values())

    sample_weights = []
    for i in train_idx:
        lbl = int(df.iloc[i]["intent_label"])
        w   = max_count / label_counts[lbl]   # inverse frequency weight
        sample_weights.append(w)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_idx),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )

    # ── Test loader from test.csv ─────────────────────────────────────────────
    test_ds = CachedDataset(test_csv, transform=eval_transform)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )

    print(f"\nDataLoader split — train: {len(train_ds)} | "
          f"val: {len(val_ds)} | test: {len(test_ds)}")

    return train_loader, val_loader, test_loader
