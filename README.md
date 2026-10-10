# Set Master

This project is about the game of set, a card game where players aim to identify a "set" of three cards based on their shapes, colors, numbers, and shading.

## Features

- Synthetic dataset generation for training and testing
- Implementation of algorithms to identify sets
- Visualization tools to display cards and sets

## Background

### Game Rules

The game consists of a deck of 81 unique cards. Each card has four features:
1. Shape: Oval, Squiggle, or Diamond
2. Color: Red, Green, or Purple
3. Number: One, Two, or Three shapes on the card
4. Shading: Solid, Striped, or Open

A "set" consists of three cards in which, for each feature, the values are either all the same or all different.
For example, three cards that are all red, have different shapes, and different numbers would form a set.

### How to Play
1. The game starts with 12 cards laid out on the table.
2. Players take turns identifying sets among the cards on the table.
3. When a player identifies a set, they call it out and point to the three cards.
4. If the set is valid, the player collects the cards, and three new cards are dealt from the deck.
5. If no sets are found among the 12 cards, three additional cards are dealt.
6. The game continues until the deck is exhausted and no more sets can be found on the table.

## First-stage card detector

The training pipeline uses PyTorch Lightning and frozen DINOv3 ConvNeXt Tiny
features. A trainable feature pyramid, Faster R-CNN proposal network, and ROI box
head predict one foreground class, `card`. This is a **frozen-backbone detection
probe**, rather than a strict linear probe. Only those detection modules enter
the optimizer; the pretrained encoder stays in evaluation mode with no gradients.

Install the optional training dependencies:

```sh
uv sync --extra training
```

Initialize the pinned DINOv3 source submodule after cloning:

```sh
git submodule update --init --recursive
```

The defaults load these repository files without downloading a model:

- Source: `src/dinov3/` (official repository, pinned as a Git submodule)
- Weights: `models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` (gitignored)

Use `--dinov3-repo` and `--weights` to override them for training. The existing
local checkpoint has been copied into `models/`; on another checkout, provide it
separately. Prediction needs the source submodule and the trained Lightning
checkpoint; it does not need the original pretrained weight file.

Generate data with `notebooks/generate_synthetic_dataset.ipynb`, then point the
trainer to a generated run directory containing `annotations.json` and `images/`:

```sh
uv run --extra training scripts/train_card_detector.py \
  --train-data data/synthetic_datasets/YOUR_RUN \
  --batch-size 2 --epochs 20
```

Without `--val-data`, the trainer makes a reproducible 80/20 image split. Supply
a separate COCO root with `--val-data` to use an independent dataset instead;
generate that dataset with a different seed. Keep real photographs from the
same capture session together when making real-data splits.

Check one training and validation batch before a full run:

```sh
uv run --extra training scripts/train_card_detector.py \
  --train-data data/synthetic_datasets/YOUR_RUN \
  --num-workers 0 --fast-dev-run
```

For a quick CPU smoke check, add `--accelerator cpu --min-size 128 --max-size 256`.
These smaller dimensions are for checking the pipeline. Normal training uses
640/1280, preserves aspect ratio, and pads batches to multiples of 32. Images
enter as RGB floats in [0, 1]; the detector applies ImageNet normalization once.
COCO `[x, y, width, height]` boxes are converted to torchvision's `xyxy` format.
Label 0 is background, and label 1 is a card. Empty images are supported.

Each run under `runs/card_detector/version_N/` records configuration, split image
IDs, hyperparameters, CSV losses, and validation box mAP (IoU 0.50:0.95), AP50,
AP75, and recall. Checkpoints contain the best validation model and the last
training state. Resume with `--resume PATH_TO_LAST_CKPT`. `--fast-dev-run` skips
normal metric logging and checkpoint saving. Start with 32-bit precision;
`--precision bf16-mixed` is available for compatible hardware.

Predict boxes, draw a debug image, and save original-resolution crops for stage two:

```sh
uv run --extra training scripts/predict_card_detector.py \
  --checkpoint runs/card_detector/version_0/checkpoints/last.ckpt \
  --image images/PXL_20230205_150413474.jpg \
  --output-dir runs/card_predictions --device cpu
```

This writes `predictions.json`, `debug.jpg`, and numbered PNG crops under `crops/`.
The default score threshold is 0.5; use `--score-threshold` to change it. Box
coordinates refer to the input image, before detector resizing. These crops are
axis-aligned and still contain rotation and perspective; stage two should be
trained on comparable crops, or use a separate rectification step later.

Evaluate on held-out real photos before judging detection quality. The current
gray synthetic backgrounds and flat lighting make synthetic validation easier
than the intended camera images.
