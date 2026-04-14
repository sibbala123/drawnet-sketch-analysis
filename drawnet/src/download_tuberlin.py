"""
DrawNet — TU-Berlin Selective Downloader
-----------------------------------------
Downloads the full TU-Berlin zip from Kaggle, extracts only the 20 categories
DrawNet needs, then deletes the zip to save disk space.

Requirements
------------
    pip install kaggle
    Place your kaggle.json API token at ~/.kaggle/kaggle.json
    (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)

Usage
-----
    cd drawnet/
    python src/download_tuberlin.py                     # dry-run: list zip contents only
    python src/download_tuberlin.py --extract           # download + extract wanted categories
    python src/download_tuberlin.py --extract --keep_zip  # keep the zip after extraction

Output
------
    data/raw/tuberlin/{category}/  ← one folder per selected category
"""

import argparse
import pathlib
import zipfile
import subprocess
import sys
import shutil

# ── Categories we want from TU-Berlin ─────────────────────────────────────────
# These are our best-guess names. The script will fuzzy-match them against
# the actual zip contents and report any mismatches so you can correct the list.
WANTED_CATEGORIES = [
    "airplane",
    "butterfly",
    "elephant",
    "horse",
    "rabbit",
    "bear (animal)",
    "mushroom",
    "cup",
    "umbrella",
    "shoe",
    "guitar",
    "snake",
    "spider",
    "apple",
    "sun",
    "bridge",
    "castle",
    "palm tree",   # TU-Berlin uses spaces in folder names
    "eye",
    "chair",
]

KAGGLE_DATASET = "zara2099/tu-berlin-hand-sketch-image-dataset"
ZIP_NAME       = "tu-berlin-hand-sketch-image-dataset.zip"
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_zip(out_dir: pathlib.Path):
    """Download the dataset zip via Kaggle CLI into out_dir."""
    print(f"Downloading {KAGGLE_DATASET} …")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(out_dir)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\nKaggle CLI failed. Make sure:\n"
              "  1. kaggle is installed: pip install kaggle\n"
              "  2. ~/.kaggle/kaggle.json exists with your API token\n"
              "  3. You have accepted the dataset terms on kaggle.com")
        sys.exit(1)
    zip_path = out_dir / ZIP_NAME
    if not zip_path.exists():
        # Kaggle sometimes names the file differently — find any zip
        zips = list(out_dir.glob("*.zip"))
        if not zips:
            print("Download appeared to succeed but no zip found in", out_dir)
            sys.exit(1)
        zip_path = zips[0]
    print(f"Downloaded: {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def list_zip_categories(zip_path: pathlib.Path) -> dict:
    """
    Inspect the zip and return a dict mapping lowercase category name → list of
    member paths inside the zip that belong to that category.
    Only looks at image files.
    """
    category_map: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            p = pathlib.PurePosixPath(name)
            if p.suffix.lower() not in IMG_EXTS:
                continue
            # Expect structure: {anything}/{category}/{filename}
            # Take the second-to-last path component as category
            parts = p.parts
            if len(parts) < 2:
                continue
            cat = parts[-2]
            category_map.setdefault(cat.lower(), []).append(name)
    return category_map


def match_wanted(category_map: dict, wanted: list[str]) -> dict:
    """
    Match wanted category names against actual zip categories.
    Returns dict: wanted_name → actual_zip_key (or None if not found).
    """
    available = set(category_map.keys())
    matches = {}
    for w in wanted:
        key = w.lower()
        if key in available:
            matches[w] = key
        else:
            # Try underscore ↔ space swap
            alt = key.replace(" ", "_") if " " in key else key.replace("_", " ")
            if alt in available:
                matches[w] = alt
            else:
                matches[w] = None
    return matches


def extract_categories(zip_path: pathlib.Path, category_map: dict,
                        matches: dict, out_dir: pathlib.Path):
    """Extract only the matched categories from the zip into out_dir."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for wanted_name, zip_key in matches.items():
            if zip_key is None:
                print(f"  [SKIP] '{wanted_name}' — not found in zip")
                continue

            # Use the canonical name from WANTED_CATEGORIES as the folder name
            # (replace spaces with underscores for filesystem safety)
            folder_name = wanted_name.replace(" ", "_")
            dest = out_dir / folder_name
            dest.mkdir(parents=True, exist_ok=True)

            members = category_map[zip_key]
            print(f"  Extracting '{zip_key}' -> {dest}  ({len(members)} images)")
            for member in members:
                filename = pathlib.PurePosixPath(member).name
                data = zf.read(member)
                (dest / filename).write_bytes(data)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",      default="data/raw/tuberlin",
                        help="Destination directory for extracted images")
    parser.add_argument("--zip_dir",  default="data/raw/tuberlin",
                        help="Where to save the downloaded zip")
    parser.add_argument("--extract",  action="store_true",
                        help="Actually download and extract (default: dry-run only)")
    parser.add_argument("--keep_zip", action="store_true",
                        help="Keep the zip file after extraction")
    parser.add_argument("--zip_path", default=None,
                        help="Path to an already-downloaded zip (skips download)")
    args = parser.parse_args()

    out_dir  = pathlib.Path(args.out)
    zip_dir  = pathlib.Path(args.zip_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)

    # ── Locate or download zip ─────────────────────────────────────────────────
    if args.zip_path:
        zip_path = pathlib.Path(args.zip_path)
        if not zip_path.exists():
            print(f"Provided zip not found: {zip_path}")
            sys.exit(1)
    else:
        zip_path = zip_dir / ZIP_NAME
        if not zip_path.exists():
            if not args.extract:
                print("Zip not found locally. Run with --extract to download it.")
                print(f"Expected: {zip_path}")
                sys.exit(0)
            zip_path = download_zip(zip_dir)

    # ── Inspect zip ────────────────────────────────────────────────────────────
    print(f"\nInspecting zip: {zip_path}")
    category_map = list_zip_categories(zip_path)
    print(f"Found {len(category_map)} categories in zip.\n")

    matches = match_wanted(category_map, WANTED_CATEGORIES)

    print("Category match report:")
    print(f"  {'Wanted':<25}  {'Zip key':<25}  Status")
    print(f"  {'-'*25}  {'-'*25}  ------")
    for w, key in matches.items():
        status = "OK" if key else "NOT FOUND"
        print(f"  {w:<25}  {str(key):<25}  {status}")

    not_found = [w for w, k in matches.items() if k is None]
    if not_found:
        print(f"\nWARNING: {len(not_found)} categories not matched.")
        print("Edit WANTED_CATEGORIES in this script to use the exact zip folder names.")
        print("\nAll available categories in zip:")
        for cat in sorted(category_map.keys()):
            print(f"  {cat}  ({len(category_map[cat])} images)")

    if not args.extract:
        print("\nDry-run complete. Run with --extract to download and extract.")
        return

    # ── Extract ────────────────────────────────────────────────────────────────
    print(f"\nExtracting to {out_dir} …")
    extract_categories(zip_path, category_map, matches, out_dir)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.keep_zip:
        zip_path.unlink()
        print(f"\nDeleted zip: {zip_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n--- Extraction summary ---")
    total = 0
    for folder in sorted(out_dir.iterdir()):
        if not folder.is_dir():
            continue
        n = sum(1 for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS)
        total += n
        print(f"  {folder.name:<25}: {n} images")
    print(f"  {'TOTAL':<25}: {total} images")
    print(f"\nDone. Images in: {out_dir}")


if __name__ == "__main__":
    main()
