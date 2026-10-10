# Set Master

Synthetic SET card generation and a two-stage vision pipeline: locate cards, then recognize their features.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [tweak_features](notebooks/tweak_features.ipynb) | Adjust card geometry, symbols, and shading |
| [generate_synthetic_dataset](notebooks/generate_synthetic_dataset.ipynb) | Generate images, binary masks, and COCO annotations |
| [train_card_detector](notebooks/train_card_detector.ipynb) | Train and inspect the card detector |
| [evaluate_card_detector](notebooks/evaluate_card_detector.ipynb) | Compare box metrics and prediction overlays on synthetic and real datasets |

## Training

Stage one uses frozen DINOv3 ConvNeXt Tiny features with a trainable FPN/Faster R-CNN head. It predicts one class, `card`.

PyTorch uses CUDA 12.6 wheels for compatibility with the GTX 1070 Ti. The training notebook requests one GPU with 32-bit precision.

```sh
uv sync --extra training
git submodule update --init --recursive
```

DINOv3 source is pinned at `src/dinov3/`. Pretrained weights belong in
`models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` and are gitignored.

In the training notebook, select a generated dataset and set `fast_dev_run = False` for full training. The default validation split is 20%; `val_data` selects a separate dataset.

Runs under `runs/card_detector/` contain configuration, split IDs, CSV metrics, and best/last checkpoints. Validation reports box mAP, AP50, AP75, and recall. Set `resume_checkpoint` to resume a run.

### CLI

```sh
uv run --extra training scripts/train_card_detector.py \
  --train-data data/synthetic_datasets/YOUR_RUN

uv run --extra training scripts/predict_card_detector.py \
  --checkpoint runs/card_detector/version_0/checkpoints/last.ckpt \
  --image images/000008.jpg
```

Prediction writes `predictions.json`, `debug.jpg`, and card crops to `runs/card_predictions/`. Trained checkpoints include the encoder weights; prediction does not need the original pretrained file.

## Reference test set

`data/testset/` contains the 13 reference photos with numbered filenames, COCO
annotations, binary masks, instance masks, and debug overlays. The current labels
cover the 12 fully visible face-up cards in each photo. Deck tops and partial
cards are excluded. `manifest.json` preserves source filenames, scene groups,
and the relationship between the original photo and its edited crop.

Editable polygons are stored in `images/reference_cards.json`. Regenerate
the ignored dataset with:

```sh
uv run scripts/export_reference_testset.py
```

## Visual tests

```sh
uv run pytest tests/visuals
```

Generated outputs overwrite `tests/output/cards/` and `tests/output/dataset_example/`.
