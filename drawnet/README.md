# DrawNet

**Multi-task CNN for joint intent recognition and deviation detection in children's freehand sketches.**

DrawNet uses a shared ResNet-50 backbone with two task-specific heads trained jointly via multi-task learning:

| Task | Head | Loss |
|---|---|---|
| Intent Recognition | 50-class softmax | CrossEntropyLoss |
| Deviation Detection | 6-class sigmoid | BCEWithLogitsLoss |

## Deviation Classes

| Index | Class | Clinical Relevance |
|---|---|---|
| 0 | rotation | spatial orientation deficits |
| 1 | closure_failure | visuomotor integration |
| 2 | perseveration | executive function / ASD |
| 3 | spatial_disorganization | planning deficits |
| 4 | size_distortion | proportion judgment |
| 5 | omission | attention / working memory |

## Project Structure

```
drawnet/
├── data/
│   ├── raw/          quickdraw/, obget/, dap/, kaggle_children/
│   ├── processed/    intent/, deviation/
│   └── augmented/
├── notebooks/
│   ├── 01_quickdraw_exploration.ipynb
│   ├── 02_clinical_data_exploration.ipynb
│   └── 03_augmentation_preview.ipynb
├── src/
│   ├── dataset.py    QuickDrawDataset, DeviationDataset, DrawNetDataset
│   ├── model.py      DrawNet (ResNet-50 backbone + two heads)
│   ├── train.py      multi-task training loop
│   ├── evaluate.py   per-task metrics
│   ├── augment.py    synthetic deviation augmentation
│   └── utils.py      checkpointing, seeding, visualisation
├── configs/
│   └── config.yaml
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── results/
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Kaggle credentials (for clinical datasets)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 3. Explore data
jupyter notebook notebooks/01_quickdraw_exploration.ipynb

# 4. Train
python src/train.py --config configs/config.yaml
```

## Data Sources

| Dataset | Source | Purpose |
|---|---|---|
| QuickDraw | Google (numpy_bitmap API) | Intent labels + clean drawings |
| Draw-a-Person (DAP) | Kaggle: lachin007/drawaperson-handdrawn-sketches-by-children | Clinical drawings |
| Children Drawings | Kaggle: vishmiperera/children-drawings | Supplemental clinical data |
