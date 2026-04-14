"""
DrawNet — Stratified Train/Test Splitter
------------------------------------------
Reads data/augmented/index.csv and produces a stratified 80/20 train/test split.

Stratification is by intent_label so every category is proportionally
represented in both splits. The ~20% deviated ratio is also naturally
preserved in both splits since deviation flags are independent of category.

Output
------
    data/augmented/train.csv   — 80% of index.csv
    data/augmented/test.csv    — 20% of index.csv

Usage
-----
    cd drawnet/
    python src/split_dataset.py
    python src/split_dataset.py --test_frac 0.2 --seed 42 --show_stats
"""

import argparse
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

DEVIATION_CLASSES = [
    "rotation",
    "closure_failure",
    "spatial_disorganization",
    "size_distortion",
]


def split(
    index_csv: str,
    out_dir:   pathlib.Path,
    test_frac: float = 0.20,
    seed:      int   = 42,
) -> tuple:
    df = pd.read_csv(index_csv)
    print(f"Loaded {len(df)} rows from {index_csv}")

    train_df, test_df = train_test_split(
        df,
        test_size    = test_frac,
        random_state = seed,
        stratify     = df["intent_label"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    train_path = out_dir / "train.csv"
    test_path  = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"Saved train.csv: {len(train_df)} rows  -> {train_path}")
    print(f"Saved test.csv : {len(test_df)} rows  -> {test_path}")

    return train_df, test_df


def print_stats(train_df: pd.DataFrame, test_df: pd.DataFrame):
    total = len(train_df) + len(test_df)
    print(f"\n{'='*60}")
    print(f"SPLIT STATISTICS  (total: {total})")
    print(f"{'='*60}")

    for name, df in [("Train", train_df), ("Test", test_df)]:
        n       = len(df)
        n_dev   = df[DEVIATION_CLASSES].any(axis=1).sum()
        n_clean = n - n_dev
        print(f"\n{name} ({n} samples):")
        print(f"  Clean   : {n_clean:6d}  ({100*n_clean/n:.1f}%)")
        print(f"  Deviated: {n_dev:6d}  ({100*n_dev/n:.1f}%)")

        print(f"  Deviation rates:")
        for col in DEVIATION_CLASSES:
            c = int(df[col].sum())
            print(f"    {col:<30s}: {c:5d}  ({100*c/n:.1f}%)")

        print(f"  Samples per intent class:")
        for label, group in df.groupby("intent_label"):
            cat = group["category"].iloc[0]
            print(f"    {int(label):2d}  {cat:<20s}: {len(group)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv",  default="data/augmented/index.csv")
    parser.add_argument("--out_dir",    default="data/augmented")
    parser.add_argument("--test_frac",  type=float, default=0.20)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--show_stats", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(args.index_csv).exists():
        print(f"index.csv not found: {args.index_csv}")
        print("Run cache_dataset.py first.")
        import sys; sys.exit(1)

    train_df, test_df = split(args.index_csv, out_dir, args.test_frac, args.seed)

    if args.show_stats:
        print_stats(train_df, test_df)
    else:
        for name, df in [("Train", train_df), ("Test", test_df)]:
            n_dev = df[DEVIATION_CLASSES].any(axis=1).sum()
            print(f"  {name}: {len(df)} samples, "
                  f"{n_dev} deviated ({100*n_dev/len(df):.1f}%)")


if __name__ == "__main__":
    main()
