import json
from collections import defaultdict
from pathlib import Path

import cv2
import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


DetectionTarget = dict[str, Tensor]
DetectionBatch = tuple[list[Tensor], list[DetectionTarget]]


class CocoCardDataset(Dataset[tuple[Tensor, DetectionTarget]]):
    """Read whole-card boxes; RGB conversion and normalization happen separately."""

    def __init__(self, root: Path) -> None:
        self.root = root
        coco = json.loads((root / "annotations.json").read_text())
        card_categories = {item["id"] for item in coco["categories"] if item["name"] == "card"}
        if not card_categories:
            raise ValueError(f"No 'card' category in {root / 'annotations.json'}")
        self.images = sorted(coco["images"], key=lambda item: item["id"])
        self.annotations = defaultdict(list)
        for annotation in coco["annotations"]:
            if annotation["category_id"] in card_categories:
                self.annotations[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        info = self.images[index]
        path = self.root / info["file_name"]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read {path}")
        height, width = image.shape[:2]
        boxes, areas, crowds = [], [], []
        for annotation in self.annotations[info["id"]]:
            x, y, w, h = annotation["bbox"]
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError(f"Invalid card box {annotation['bbox']} in {path}")
            boxes.append([x, y, x + w, y + h])
            areas.append(annotation.get("area", w * h))
            crowds.append(annotation.get("iscrowd", 0))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(info["id"], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        }
        return tensor, target


def collate_detection(batch: list[tuple[Tensor, DetectionTarget]]) -> DetectionBatch:
    # Detection keeps variable image sizes and object counts rather than stacking targets.
    return [item[0] for item in batch], [item[1] for item in batch]


class CardDataModule(pl.LightningDataModule):
    def __init__(
        self, train_root: Path, val_root: Path | None = None, batch_size: int = 2,
        num_workers: int = 4, val_fraction: float = 0.2, seed: int = 42,
    ) -> None:
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        dataset = CocoCardDataset(self.train_root)
        if self.val_root is not None:
            if self.val_root.resolve() == self.train_root.resolve():
                raise ValueError("Training and validation roots must differ")
            self.train_dataset = dataset
            self.val_dataset = CocoCardDataset(self.val_root)
            train_indices = list(range(len(dataset)))
            val_indices = list(range(len(self.val_dataset)))
            val_source = self.val_dataset
        else:
            if len(dataset) < 2 or not 0 < self.val_fraction < 1:
                raise ValueError("Splitting requires at least two images and 0 < val_fraction < 1")
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(self.seed)).tolist()
            val_count = min(len(dataset) - 1, max(1, round(len(dataset) * self.val_fraction)))
            val_indices, train_indices = indices[:val_count], indices[val_count:]
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
            val_source = dataset
        if not len(self.train_dataset) or not len(self.val_dataset):
            raise ValueError("Training and validation datasets must both contain images")
        self.split_manifest = {
            "seed": self.seed,
            "train_root": str(self.train_root.resolve()),
            "val_root": str((self.val_root or self.train_root).resolve()),
            "train_image_ids": [dataset.images[i]["id"] for i in train_indices],
            "val_image_ids": [val_source.images[i]["id"] for i in val_indices],
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_detection,
            persistent_workers=self.num_workers > 0,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=collate_detection, persistent_workers=self.num_workers > 0,
        )
