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

Stage one uses frozen DINOv3 ConvNeXt Tiny features and a trainable FPN. All variants predict one class, `card`:

| Variant | Head | Geometry |
| --- | --- | --- |
| `box` | Faster R-CNN | Axis-aligned boxes |
| `oriented` | Dense FPN head with rotated NMS | Center, short width, long height, angle |
| `corners` | Keypoint R-CNN | Four corner heatmaps and axis-aligned boxes |

Shared code lives in `training/backbone.py`, `data.py`, `config.py`, and
`evaluation.py`; variant implementations live in `training/models/`.

PyTorch uses CUDA 12.6 wheels for compatibility with the GTX 1070 Ti. The training notebook requests one GPU with 32-bit precision.

```sh
uv sync --extra training
git submodule update --init --recursive
```

DINOv3 source is pinned at `src/dinov3/`. Pretrained weights belong in
`models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` and are gitignored.

In the training notebook, select a generated dataset and set `fast_dev_run = False` for full training. The default validation split is 20%; `val_data` selects a separate dataset.

Select `variant` in the training notebook. Experiment names are generated from
the detector type and timestamp, such as `corners_20261010_133100_123456`. Runs under
`runs/card_detector/<variant>/<experiment>/version_N/` contain configuration,
split IDs, CSV metrics, and best/last checkpoints. Geometry variants select the
best checkpoint by geometry AP; the baseline uses box AP. Existing baseline
checkpoints remain loadable. Set `resume_checkpoint` to resume a run.

Training logs losses, learning rate, validation AP, and real-data AP,
precision/recall, and geometry metrics to CSV and TensorBoard. The notebook
includes an embedded TensorBoard dashboard. A separate local dashboard can be
opened with:

```sh
uv run --extra training tensorboard --logdir runs/card_detector
```

`test_data` defaults to `data/testset/` and is evaluated after every epoch.
It is excluded from training, early stopping, and checkpoint selection.
Early stopping monitors validation AP with patience 5 and minimum improvement
0.001; both are configurable. Validation overlays reuse the existing validation
predictions. All validation and real images are saved by default; set explicit
overlay ID lists to limit them. Epoch outputs live in
`<run>/epoch_evaluations/epoch_0001/`, with `validation/`, `real/`, and
`metrics.json`. TensorBoard images can be compared across epochs.

Evaluation loads the variant recorded in the selected checkpoint. All variants
report box metrics; geometry variants also report rotated/polygon AP. Corners
report mean corner error divided by the target diagonal, on polygon-IoU matches
of at least 0.5. These are custom single-class geometry metrics, not COCO OKS.
Overlays and rectified crops are saved for selected images. Evaluation folders
include the variant, checkpoint experiment/run, checkpoint name, and dataset run.

## Geometry annotations

New synthetic datasets include exact transformed corners, polygon segmentation,
and fitted oriented boxes. Rotation defaults to the full `[-180°, 180°]` range.
Horizontal and vertical skew default to `[-15°, 15°]`; both skew ranges and
perspective jitter are configurable in the generator notebook. A shared
homography transforms pixels, masks, and corners without clipping the card.
Debug images show green axis boxes, blue oriented boxes, orange corners
numbered 1–4, and cyan mask outlines.

Corners run clockwise in image coordinates, starting at the smallest `x+y`.
They describe virtual sharp card corners, independent of printed orientation.
OBBs use `[cx, cy, short_width, long_height, angle_degrees]`; the width-axis angle
is clockwise in image coordinates and lies in `[-90, 90)`.

Real corners are estimated by fitting straight edges to the existing polygons
and stored in `images/reference_cards.json`. They need manual review before
using corner errors as a precise benchmark. Existing synthetic datasets can be
upgraded explicitly; their fitted corners are estimates, not original render
transforms:

```sh
uv run scripts/add_card_geometry.py --dataset data/synthetic_datasets/YOUR_RUN
```

Use the same dataset and split seed across experiments. The oriented experiment
uses a dense detection head, so comparisons also include that architecture
difference. Its OpenCV rotated suppression runs on the CPU.

### CLI

```sh
uv run --extra training scripts/train_card_detector.py \
  --train-data data/synthetic_datasets/YOUR_RUN \
  --variant corners

uv run --extra training scripts/predict_card_detector.py \
  --checkpoint runs/card_detector/corners/YOUR_EXPERIMENT/version_0/checkpoints/last.ckpt \
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
