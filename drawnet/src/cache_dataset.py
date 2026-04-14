"""
DrawNet — Dataset Caching Script
----------------------------------
Reads master_annotations.csv, applies deviation augmentation to flagged images,
saves every image as a PNG, and writes index.csv.

Run this ONCE before training. Training then reads PNGs from disk instead of
re-computing augmentation every batch (~8-10x faster on Windows).

Pipeline
--------
    build_annotations.py  ->  data/processed/master_annotations.csv
            |
            v
    cache_dataset.py       ->  data/augmented/index.csv
                               data/augmented/images/{000000..N}.png

Output CSV columns
------------------
    filepath, intent_label, category, source,
    rotation, closure_failure, spatial_disorganization, size_distortion

Usage
-----
    cd drawnet/
    python src/cache_dataset.py
    python src/cache_dataset.py --config configs/config.yaml --out data/augmented
    python src/cache_dataset.py --no_resume     # restart from scratch
"""

import argparse
import pathlib
import sys
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from augment  import generate_deviation_sample, DEVIATION_NAMES
from dataset  import numpy_bitmap_to_image

IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_quickdraw_image(filepath: str, row_index: int) -> Image.Image:
    """Load a single QuickDraw bitmap from a .npy file by row index."""
    arr    = np.load(filepath, mmap_mode="r")
    bitmap = arr[row_index]
    return numpy_bitmap_to_image(bitmap, IMAGE_SIZE)


def load_tuberlin_image(filepath: str) -> Image.Image:
    """Load a TU-Berlin JPEG and resize to IMAGE_SIZE."""
    return (Image.open(filepath)
                 .convert("RGB")
                 .resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))


def load_image(row: pd.Series) -> Image.Image:
    """Dispatch to the correct loader based on source column."""
    if row["source"] == "quickdraw":
        return load_quickdraw_image(row["filepath"], int(row["npy_row_index"]))
    else:
        return load_tuberlin_image(row["filepath"])


# ---------------------------------------------------------------------------
# Main caching loop
# ---------------------------------------------------------------------------

def cache_all(
    annotations_csv: str,
    out_dir:         pathlib.Path,
    resume:          bool = True,
):
    """
    Process every row in master_annotations.csv:
      - Load the image (QuickDraw bitmap or TU-Berlin JPEG)
      - If will_deviate=1: apply generate_deviation_sample → get deviation labels
      - If will_deviate=0: deviation labels are all zeros
      - Save as PNG
      - Append row to index.csv

    Parameters
    ----------
    annotations_csv : str
        Path to master_annotations.csv from build_annotations.py.
    out_dir : pathlib.Path
        Output directory for PNGs and index.csv.
    resume : bool
        If True and index.csv already exists, skip already-processed images.
    """
    img_dir  = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "index.csv"

    # Resume support — read already-completed rows
    if resume and csv_path.exists():
        existing      = pd.read_csv(csv_path)
        done_set      = set(existing["filepath"].tolist())
        completed_rows = existing.to_dict("records")
        print(f"Resuming — {len(completed_rows)} images already cached.")
    else:
        done_set       = set()
        completed_rows = []

    annotations = pd.read_csv(annotations_csv)
    total       = len(annotations)
    print(f"\nProcessing {total} images -> {out_dir} ...")

    for idx, row in tqdm(annotations.iterrows(), total=total, unit="img"):
        out_fname = f"{idx:06d}.png"
        out_path  = str((img_dir / out_fname).resolve())

        if out_path in done_set:
            continue

        # Load image
        try:
            img = load_image(row)
        except Exception as e:
            print(f"\n[WARN] Could not load image at index {idx}: {e} — skipping.")
            continue

        # Apply deviation augmentation if flagged
        if int(row["will_deviate"]) == 1:
            img, dev_vector = generate_deviation_sample(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
        else:
            dev_vector = [0] * len(DEVIATION_NAMES)

        img.save(out_path, format="PNG", optimize=True)

        record = {
            "filepath":     out_path,
            "intent_label": int(row["intent_label"]),
            "category":     row["category"],
            "source":       row["source"],
        }
        for name, val in zip(DEVIATION_NAMES, dev_vector):
            record[name] = val
        completed_rows.append(record)

        # Checkpoint to CSV every 2000 images
        if len(completed_rows) % 2000 == 0:
            pd.DataFrame(completed_rows).to_csv(csv_path, index=False)

    # Final save
    df = pd.DataFrame(completed_rows)
    df.to_csv(csv_path, index=False)

    # Summary
    total_saved = len(df)
    n_deviated  = df[DEVIATION_NAMES].any(axis=1).sum()
    print(f"\nDone. {total_saved} images saved to {out_dir}")
    print(f"Disk usage estimate: {total_saved * 20 / 1024:.0f} MB")

    print("\n--- Deviation distribution ---")
    for col in DEVIATION_NAMES:
        n = int(df[col].sum())
        print(f"  {col:30s}: {n:6d}  ({100*n/total_saved:.1f}%)")
    print(f"  {'Any deviation':<30s}: {n_deviated:6d}  "
          f"({100*n_deviated/total_saved:.1f}%)")

    print("\n--- Intent class distribution ---")
    for label, group in df.groupby("intent_label"):
        cat = group["category"].iloc[0]
        print(f"  {int(label):2d}  {cat:<20s}: {len(group):5d}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--annotations", default=None,
                        help="Path to master_annotations.csv "
                             "(defaults to data.annotations_csv in config)")
    parser.add_argument("--out",         default="data/augmented")
    parser.add_argument("--no_resume",   action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    annotations_csv = args.annotations or cfg["data"]["annotations_csv"]

    if not pathlib.Path(annotations_csv).exists():
        print(f"Annotations file not found: {annotations_csv}")
        print("Run build_annotations.py first.")
        sys.exit(1)

    cache_all(
        annotations_csv = annotations_csv,
        out_dir         = pathlib.Path(args.out),
        resume          = not args.no_resume,
    )


if __name__ == "__main__":
    main()
