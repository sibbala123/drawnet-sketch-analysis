"""
DrawNet — Annotation Builder
------------------------------
Scans QuickDraw .npy files and TU-Berlin image folders, assigns intent labels
from category names, and randomly flags 20% of the combined pool as deviated.

This script produces a single master_annotations.csv that is the input to
cache_dataset.py. It does NOT apply any augmentation — it only decides which
images will receive deviations during caching.

Output columns
--------------
    source          : "quickdraw" or "tuberlin"
    filepath        : absolute path to image file (tuberlin) or .npy file (quickdraw)
    npy_row_index   : row index within .npy file (quickdraw only, else -1)
    intent_label    : integer class index (0-29)
    category        : human-readable category name
    will_deviate    : 1 if this image will receive deviation augmentation, else 0

Usage
-----
    cd drawnet/
    python src/build_annotations.py
    python src/build_annotations.py --config configs/config.yaml --show_stats
"""

import argparse
import pathlib
import random
import yaml
import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

def scan_quickdraw(
    data_dir: pathlib.Path,
    categories: list,
    samples_per_class: int,
    label_offset: int = 0,
) -> list:
    """
    Build a row per QuickDraw sample.
    filepath = path to .npy file; npy_row_index = row within that file.
    """
    rows = []
    for i, cat in enumerate(categories):
        npy_path = data_dir / f"{cat}.npy"
        if not npy_path.exists():
            print(f"[QuickDraw] WARNING: {npy_path} not found — skipping.")
            continue
        arr = np.load(npy_path, mmap_mode="r")
        n   = min(samples_per_class, len(arr))
        label = i + label_offset
        for row_idx in range(n):
            rows.append({
                "source":        "quickdraw",
                "filepath":      str(npy_path.resolve()),
                "npy_row_index": row_idx,
                "intent_label":  label,
                "category":      cat,
                "will_deviate":  0,   # filled later
            })
        print(f"  [QuickDraw] {cat:<20s}: {n} samples  (label {label})")
    return rows


def scan_tuberlin(
    data_dir: pathlib.Path,
    categories: list,
    label_offset: int = 10,
) -> list:
    """
    Build a row per TU-Berlin image file.
    filepath = absolute path to image; npy_row_index = -1.
    """
    rows = []
    for i, cat in enumerate(categories):
        cat_dir = data_dir / cat
        if not cat_dir.exists():
            print(f"[TU-Berlin] WARNING: {cat_dir} not found — skipping.")
            continue
        label  = i + label_offset
        images = [p for p in cat_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        for img_path in images:
            rows.append({
                "source":        "tuberlin",
                "filepath":      str(img_path.resolve()),
                "npy_row_index": -1,
                "intent_label":  label,
                "category":      cat,
                "will_deviate":  0,
            })
        print(f"  [TU-Berlin] {cat:<20s}: {len(images)} samples  (label {label})")
    return rows


# ---------------------------------------------------------------------------
# Deviation flag assignment
# ---------------------------------------------------------------------------

def assign_deviation_flags(rows: list, deviation_fraction: float, seed: int = 42) -> list:
    """
    Randomly flag `deviation_fraction` of the combined pool as will_deviate=1.
    The selection is stratified — each category contributes its proportional share
    of deviated images so no single category dominates the deviated pool.
    """
    rng = random.Random(seed)
    df  = pd.DataFrame(rows)

    deviated_indices = set()
    for cat, group in df.groupby("category"):
        n_deviate = max(1, round(len(group) * deviation_fraction))
        chosen    = rng.sample(list(group.index), n_deviate)
        deviated_indices.update(chosen)

    for i in deviated_indices:
        rows[i]["will_deviate"] = 1

    return rows


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def print_stats(df: pd.DataFrame):
    total    = len(df)
    n_dev    = df["will_deviate"].sum()
    n_clean  = total - n_dev

    print(f"\n{'='*60}")
    print(f"ANNOTATION STATISTICS  (total: {total})")
    print(f"{'='*60}")
    print(f"  Clean (will_deviate=0) : {n_clean:6d}  ({100*n_clean/total:.1f}%)")
    print(f"  Deviated (will_deviate=1): {n_dev:6d}  ({100*n_dev/total:.1f}%)")

    print("\n--- Per source ---")
    print(df.groupby("source")["will_deviate"]
            .agg(total="count", deviated="sum")
            .assign(pct=lambda x: (100 * x["deviated"] / x["total"]).round(1))
            .to_string())

    print("\n--- Per category (intent_label, count, n_deviated) ---")
    summary = (df.groupby(["intent_label", "category"])
                 .agg(total=("will_deviate", "count"),
                      deviated=("will_deviate", "sum"))
                 .reset_index()
                 .sort_values("intent_label"))
    for _, row in summary.iterrows():
        print(f"  {int(row['intent_label']):2d}  {row['category']:<20s}: "
              f"{int(row['total']):5d} total, {int(row['deviated']):4d} deviated")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--out",         default="data/processed/master_annotations.csv")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--show_stats",  action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    qd_dir       = pathlib.Path(cfg["data"]["quickdraw_dir"])
    tb_dir       = pathlib.Path(cfg["data"]["tuberlin_dir"])
    out_path     = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qd_categories = cfg["quickdraw_categories"]
    tb_categories = cfg["tuberlin_categories"]
    samples_pc    = cfg["data"]["quickdraw_samples_per_class"]
    dev_fraction  = cfg["data"]["deviation_fraction"]

    print("Scanning QuickDraw ...")
    qd_rows = scan_quickdraw(qd_dir, qd_categories, samples_pc, label_offset=0)

    print("\nScanning TU-Berlin ...")
    tb_rows = scan_tuberlin(tb_dir, tb_categories, label_offset=len(qd_categories))

    all_rows = qd_rows + tb_rows
    print(f"\nTotal samples before deviation flagging: {len(all_rows)}")

    print(f"\nAssigning deviation flags ({dev_fraction*100:.0f}% of each category) ...")
    all_rows = assign_deviation_flags(all_rows, dev_fraction, seed=args.seed)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    if args.show_stats:
        print_stats(df)
    else:
        n_dev = df["will_deviate"].sum()
        print(f"  Clean: {len(df) - n_dev}  |  Deviated: {n_dev}  "
              f"({100*n_dev/len(df):.1f}%)")


if __name__ == "__main__":
    main()
