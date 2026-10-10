# Set Master

Synthetic SET card generation and detection with a frozen DINOv3 ConvNeXt Tiny backbone.

## Notebooks

Use the notebooks for configuration, previews, and workflow documentation.

| Notebook | Purpose |
| --- | --- |
| [Tweak features](notebooks/tweak_features.ipynb) | Adjust card geometry, symbols, and shading |
| [Download backgrounds](notebooks/download_backgrounds.ipynb) | Download and review real tabletop backgrounds |
| [Generate synthetic dataset](notebooks/generate_synthetic_dataset.ipynb) | Generate images, masks, and COCO annotations |
| [Augmentation benchmark](notebooks/benchmark_augmentations.ipynb) | Tune lighting and camera effects with paired previews |
| [Train card detector](notebooks/train_card_detector.ipynb) | Train box, oriented-box, corner heatmap, or quadrilateral regression detectors |
| [Evaluate card detector](notebooks/evaluate_card_detector.ipynb) | Evaluate synthetic and real data with prediction overlays |

## Setup

```sh
uv sync --extra training
git submodule update --init --recursive
```

Select the project's `.venv` kernel. Place the pretrained weights at
`models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth`.

### Local W&B

With Docker and Compose installed:

```sh
docker compose up -d
```

Open [localhost:8080](http://localhost:8080), create a local account, and activate
it with a license from [W&B Deployer](https://deploy.wandb.ai/deploy). Then log in:

```sh
uv run --extra training wandb login --host http://localhost:8080
```

The training notebook links to the run dashboard. Server data persists across
`docker compose down`.

## Data and outputs

- `images/`: reference photos and editable annotations in `reference_cards.json`.
- `data/synthetic_datasets/`: generated dataset runs.
- `data/backgrounds/tabletops/`: downloaded backgrounds and attribution manifest.
- `data/testset/`: real evaluation data; rebuild with `uv run scripts/export_reference_testset.py`.
- `runs/`: training checkpoints, metrics, preview grids, and evaluation overlays.

The real labels cover fully visible face-up cards; deck tops and partial cards
are excluded. Generated data, runs, and pretrained weights are gitignored.

Corner and quadrilateral targets use `corner_order: "card_relative"`: clockwise
vertices with edges 0–1 and 2–3 corresponding to the card's short sides, regardless
of their projected lengths. Training accepts a 180° rotation of this order but
penalizes 90° rotations. Augmentation preserves the short-side identity, and crops
use the predicted order directly. Retrain models previously trained on image-relative
corner labels before using their predictions for orientation.

New synthetic datasets preserve the original sprite corner identities through all
render transforms. To migrate older synthetic runs with saved transforms:

```sh
uv run scripts/migrate_synthetic_orientation.py --dataset data/synthetic_datasets/RUN_NAME
```

The real source annotations store visually reviewed card-relative corners. Exporting
the testset preserves that order; geometry fitting alone cannot provide these labels.

## Visual tests

```sh
uv run pytest tests/visuals
```

Outputs overwrite `tests/output/cards/` and `tests/output/dataset_example/`.
