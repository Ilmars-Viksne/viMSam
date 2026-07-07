# viMSam Trainer

Fine-tuning workflows for micro-SAM models on microscopy images and raw frames.

## Install

From the `trainer_app` directory:

```bash
pip install -e .
```

For training, install the ML extra or use a conda/mamba environment that provides
`micro_sam`, `torch`, and `torchvision`:

```bash
pip install -e ".[ml]"
```

## CLI

### Raw Frames

```bash
vimsam-trainer \
  --images data/training/raw_frames \
  --masks data/training/masks \
  --out data/models/cell_model_v1 \
  --workflow raw_frames \
  --model vit_b \
  --epochs 50 \
  --batch-size 2 \
  --raw-width 1024 \
  --raw-height 1024 \
  --preprocessing-method fixed_16bit \
  --patch-shape 512,512 \
  --device cuda
```

### Standard Images

```bash
vimsam-trainer \
  --images data/training/images \
  --masks data/training/masks \
  --out data/models/cell_model_v1 \
  --workflow image_frames \
  --model vit_b \
  --epochs 50 \
  --batch-size 2 \
  --preprocessing-method percentile \
  --patch-shape 512,512 \
  --device auto
```

## Python API

```python
from vimsam_trainer import TrainerApp, TrainingConfig

result = TrainerApp().run(
    TrainingConfig(
        workflow="raw_frames",
        images_path="data/training/raw_frames",
        masks_path="data/training/masks",
        output_path="data/models/cell_model_v1",
        model_type="vit_b",
        epochs=50,
        batch_size=2,
        raw_width=1024,
        raw_height=1024,
        preprocessing_method="fixed_16bit",
        patch_shape=(512, 512),
        device="cuda",
    )
)

print(result.checkpoint_path)
```

## Expected Data Layout

```text
data/training/raw_frames/
├── 20260226213602.raw
└── 20260226213625.raw

data/training/masks/
├── 20260226213602.tif
└── 20260226213625.tif
```

Image and mask files are paired by file stem. Training masks should be 2D
instance masks, where background is `0` and each object has a positive integer
label. Binary masks can represent one object per image, but micro-SAM may reject
samples that do not contain enough valid object labels for the selected training
mode.

The trained checkpoint is written below:

```text
<out>/checkpoints/<out-folder-name>/best.pt
```
