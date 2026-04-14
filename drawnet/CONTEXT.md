# DrawNet — Full Project Context

## What is DrawNet?

DrawNet is a multi-task deep learning model for sketch analysis. Given a sketch as input, it simultaneously:

1. **Intent Recognition** — classifies the sketch into one of 30 subject categories (face, cat, airplane, etc.)
2. **Deviation Detection** — detects the presence of geometric drawing deviations across 4 classes (multi-label binary)

The deviations capture structural properties of a drawing — rotation, closure failure, spatial disorganization, and size distortion — that characterise drawing quality and geometry.

---

## Problem Statement

Sketch recognition and quality analysis are useful for applications ranging from educational tools to drawing assistance. DrawNet tackles both simultaneously via a shared backbone:

- The **intent head** identifies what the sketch depicts
- The **deviation head** flags which geometric irregularities are present

DrawNet does NOT make clinical diagnoses. It is a sketch analysis model trained on synthetic and hand-drawn sketch data.

---

## The 4 Deviation Classes

Each deviation is a binary label (present / absent):

| Deviation | Description |
|---|---|
| `rotation` | Drawing is rotated significantly from its expected orientation (>15°) |
| `closure_failure` | Open shapes that should be closed — endpoints don't connect |
| `spatial_disorganization` | Drawing components are scattered without coherent spatial structure |
| `size_distortion` | Drawing is abnormally small or large relative to the canvas |

---

## Model Architecture

**DrawNet** — shared ResNet-50 backbone with two task-specific heads:

```
Input image (224×224×3)
        ↓
ResNet-50 backbone (pretrained ImageNet)
  → strips final avgpool + FC
  → outputs (B, 2048) feature vector
        ↓
    ┌───────────────────────────────┐
    │                               │
Intent Head                  Deviation Head
Dropout(0.5)                 Dropout(0.5)
Linear(2048→512)             Linear(2048→256)
ReLU                         ReLU
Linear(512→30)               Linear(256→4)
    │                               │
Softmax (30 classes)         Sigmoid (4 binary outputs)
```

**Loss function:**
```
L_total = λ1 × CrossEntropy(intent) + λ2 × BCEWithLogitsLoss(deviation)
```
- λ1 = λ2 = 1.0 (equal weighting)
- All images have valid intent labels — no masking required
- BCEWithLogitsLoss uses per-class `pos_weight` to handle class imbalance (80% clean / 20% deviated)

**Backbone unfreezing schedule (progressive fine-tuning):**
- Epochs 1–10: backbone frozen, only heads train
- Epochs 11–20: unfreeze layer4 only
- Epochs 21–50: unfreeze full backbone at lower LR (1e-5)

---

## Datasets

### 1. QuickDraw (Google) — 10 categories
- **What:** 50,000 bitmap drawings across 10 categories from Google's QuickDraw dataset
- **Format:** `.npy` files, each row is a 784-element flat array (28×28 bitmap)
- **Categories:** face, house, tree, cat, car, bird, dog, fish, flower, bicycle
- **Samples:** 5,000 per category

### 2. TU-Berlin Hand Sketch Dataset (Kaggle) — 20 additional categories
- **What:** Hand-drawn pen-on-paper sketches across 250 categories
- **Format:** JPEG/PNG images organised by category folder
- **Selected categories (20):** airplane (434), butterfly (283), elephant (357), horse (584), rabbit (433), bear_animal (418), mushroom (701), cup (452), umbrella (1010), shoe (1214), guitar (255), snake (311), spider (1272), apple (358), sun (1528), bridge (1085), castle (1301), palm_tree (1548), eye (348), chair (266)
- **Total selected:** 14,158 images
- **Note:** Image counts vary significantly per category (255–1,548)

### Combined
- **Total:** ~64,158 images across 30 intent categories
- **Intent label imbalance:** QuickDraw categories ~5,000 images vs TU-Berlin 255–1,548 — handled via weighted sampling in training

---

## Data Pipeline

```
QuickDraw .npy files  +  TU-Berlin JPEG folders
                ↓
    build_annotations.py
        • Assigns intent labels from category names
        • Randomly flags 20% of combined pool as will_deviate=1
        • Writes data/processed/master_annotations.csv
                ↓
    cache_dataset.py
        • Reads master_annotations.csv
        • Applies augment.py to flagged images (Option A: p_each=0.6, min_deviations=1)
        • Saves PNGs to data/augmented/images/
        • Writes data/augmented/index.csv
          (filepath, intent_label, rotation, closure_failure,
           spatial_disorganization, size_distortion)
                ↓
    split_dataset.py
        • Stratified 80/20 split on index.csv (stratified by intent_label)
        • Writes data/augmented/train.csv and data/augmented/test.csv
        • Both splits preserve ~20% deviated ratio
                ↓
    train.py
        • Reads train.csv via CachedDataset
        • Internal 80/20 split → train / val
        • WeightedRandomSampler to up-sample TU-Berlin categories (fewer images)
                ↓
    evaluate.py
        • Reads test.csv
        • Reports intent top-1/top-5 accuracy, macro F1
        • Reports per-class AUROC, macro F1, Hamming loss for deviation
```

---

## Augmentation Logic (`augment.py`)

Deviations are applied only to the 20% of images flagged in `master_annotations.csv`.
Within each flagged image, each of the 4 deviations is applied independently at `p_each=0.6`,
with `min_deviations=1` to guarantee at least one deviation fires.

| Function | What it does |
|---|---|
| `apply_rotation` | Rotates 15–90° with white background fill |
| `apply_closure_failure` | Erases pixels near stroke endpoints (10% of stroke pixels) |
| `apply_spatial_disorganization` | Divides into 3×3 grid, randomly shifts cells |
| `apply_size_distortion` | Scales drawing 0.5–2× within canvas |

Expected deviation distribution within the 20% deviated subset:
- ~1.6 deviations per deviated image on average (4 × 0.6 × P(at least 1 fires))
- ~30% of deviated images will have 1 deviation
- ~35% will have 2 deviations
- ~22% will have 3 deviations
- ~13% will have all 4

---

## Intent Category Map (30 classes)

| Index | Category | Source |
|---|---|---|
| 0 | face | QuickDraw |
| 1 | house | QuickDraw |
| 2 | tree | QuickDraw |
| 3 | cat | QuickDraw |
| 4 | car | QuickDraw |
| 5 | bird | QuickDraw |
| 6 | dog | QuickDraw |
| 7 | fish | QuickDraw |
| 8 | flower | QuickDraw |
| 9 | bicycle | QuickDraw |
| 10 | airplane | TU-Berlin |
| 11 | butterfly | TU-Berlin |
| 12 | elephant | TU-Berlin |
| 13 | horse | TU-Berlin |
| 14 | rabbit | TU-Berlin |
| 15 | bear_(animal) | TU-Berlin |
| 16 | mushroom | TU-Berlin |
| 17 | cup | TU-Berlin |
| 18 | umbrella | TU-Berlin |
| 19 | shoe | TU-Berlin |
| 20 | guitar | TU-Berlin |
| 21 | snake | TU-Berlin |
| 22 | spider | TU-Berlin |
| 23 | apple | TU-Berlin |
| 24 | sun | TU-Berlin |
| 25 | bridge | TU-Berlin |
| 26 | castle | TU-Berlin |
| 27 | palm_tree | TU-Berlin |
| 28 | eye | TU-Berlin |
| 29 | chair | TU-Berlin |

*Note: Verify exact folder names in TU-Berlin after download and update this table if needed.*

---

## Training Configuration

**Local (Windows, RTX 3050 4GB):**
- batch_size: 32
- pretrained: false (SSL certificate block on Windows)
- num_workers: 2

**Colab (T4 GPU, 15GB):**
- batch_size: 64
- pretrained: true
- num_workers: 4

**Optimizer:** AdamW, lr=1e-4, weight_decay=1e-5
**Scheduler:** CosineAnnealingLR over 50 epochs
**Backbone LR:** 1e-5 (when unfrozen)

---

## File Structure

```
drawnet/
├── configs/
│   ├── config.yaml              # Local training config
│   └── config_colab.yaml        # Colab training config
├── data/
│   ├── raw/
│   │   ├── quickdraw/           # .npy files (10 categories)
│   │   └── tuberlin/            # JPEG images organised by category folder
│   ├── augmented/               # Pre-cached PNGs (generated by cache_dataset.py)
│   │   ├── index.csv            # filepath, intent_label, 4 deviation columns
│   │   ├── train.csv            # 80% stratified split (generated by split_dataset.py)
│   │   ├── test.csv             # 20% stratified split
│   │   └── images/              # 000000.png … N.png
│   └── processed/
│       └── master_annotations.csv  # filepath/index, source, intent_label, will_deviate
├── src/
│   ├── model.py                 # DrawNet architecture + DrawNetLoss
│   ├── dataset.py               # Dataset classes + DataLoader builder
│   ├── augment.py               # 4 PIL-based deviation augmentation functions
│   ├── build_annotations.py     # Scans data sources, assigns labels, flags deviated images
│   ├── cache_dataset.py         # Reads annotations, applies augmentation, saves PNGs
│   ├── split_dataset.py         # Stratified train/test split on index.csv
│   ├── train.py                 # Main training loop
│   ├── evaluate.py              # Metrics: AUROC, F1, top-1/5 accuracy
│   └── utils.py                 # Checkpoint save/load, seeding, plotting
├── notebooks/
│   └── 03_colab_training.ipynb  # End-to-end Colab pipeline
└── outputs/
    ├── checkpoints/             # best.pt, epoch_010.pt, epoch_020.pt
    ├── logs/                    # train_log.csv (per-epoch metrics)
    └── results/                 # test_metrics.csv, training_curves.png
```

---

## Evaluation Metrics

**Intent task:**
- Top-1 accuracy (primary)
- Top-5 accuracy
- Per-class macro F1

**Deviation task:**
- Per-class AUROC (primary — handles class imbalance well)
- Macro F1 at threshold 0.5
- Hamming loss
- Subset accuracy (exact match)

---

## Known Limitations

1. **Synthetic deviations only** — deviations are applied via geometric augmentation, not drawn organically
2. **Class imbalance** — QuickDraw categories have ~60× more images than TU-Berlin categories; addressed via weighted sampling
3. **Test distribution = train distribution** — both splits come from the same augmented pool; out-of-distribution generalisation is not validated
4. **TU-Berlin is adult drawings** — unlike QuickDraw's crowdsourced sketches, TU-Berlin drawings are from a controlled sketch recognition study

---

## Current Status (April 2026)

Redesigning the project from scratch:
- Dropped DAP clinical dataset (person-only, no intent diversity)
- Dropped 2 weakest deviation classes (perseveration, omission)
- Added TU-Berlin for 20 new intent categories
- New unified data pipeline: build_annotations → cache → split → train

Previous local run (epoch 20, now obsolete):
- val_loss=2.201, intent_top1=34%, dev_f1=0.800
- Trained on old 6-class deviation setup with DAP clinical data
