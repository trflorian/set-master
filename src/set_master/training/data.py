import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import cv2
import lightning.pytorch as pl
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, get_worker_info
from set_master.training.config import DetectorVariant
from set_master.training.augmentation import AugmentationConfig, CardAugmentation


DetectionTarget = dict[str, Tensor]
DetectionBatch = tuple[list[Tensor], list[DetectionTarget]]


class CocoCardDataset(Dataset[tuple[Tensor, DetectionTarget]]):
    """Read card boxes, geometry, and polygon masks for the selected task."""

    def __init__(self, root: Path, variant: DetectorVariant = DetectorVariant.BOX,
                 augmentation: CardAugmentation | None = None) -> None:
        self.root = root
        self.variant = variant
        self.augmentation = augmentation
        self.augmentation_worker_seed: int | None = None
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
        boxes, areas, crowds, keypoints, oriented_boxes = [], [], [], [], []
        for annotation in self.annotations[info["id"]]:
            x, y, w, h = annotation["bbox"]
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError(f"Invalid card box {annotation['bbox']} in {path}")
            boxes.append([x, y, x + w, y + h])
            areas.append(annotation.get("area", w * h))
            crowds.append(annotation.get("iscrowd", 0))
            if "keypoints" in annotation:
                if (self.variant in (DetectorVariant.CORNERS, DetectorVariant.QUADRILATERAL)
                        and annotation.get('corner_order') != 'card_relative'):
                    raise ValueError(f'Missing card-relative corner order in {path}; migrate orientation annotations first')
                keypoints.append(annotation["keypoints"])
            if "oriented_bbox" in annotation:
                oriented_boxes.append(annotation["oriented_bbox"])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(info["id"], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        }
        if self.variant.is_semantic or self.variant == DetectorVariant.INSTANCE:
            masks = np.zeros((len(boxes), height, width), dtype=np.uint8)
            for mask, annotation in zip(masks, self.annotations[info["id"]]):
                contours = [np.rint(np.asarray(segment).reshape(-1, 2)).astype(np.int32)
                            for segment in annotation["segmentation"]]
                if not contours:
                    raise ValueError(f"Missing card segmentation in {path}")
                cv2.fillPoly(mask, contours, 1)
            target["masks"] = torch.from_numpy(masks)
        if self.variant in (DetectorVariant.CORNERS, DetectorVariant.QUADRILATERAL) and len(keypoints) != len(boxes):
            raise ValueError(f"Missing corner targets in {path}; add geometry annotations first")
        if self.variant == DetectorVariant.ORIENTED and len(oriented_boxes) != len(boxes):
            raise ValueError(f"Missing oriented targets in {path}; add geometry annotations first")
        if len(keypoints) == len(boxes):
            target["keypoints"] = torch.tensor(keypoints, dtype=torch.float32).reshape(-1, 4, 3)
        if len(oriented_boxes) == len(boxes):
            target["oriented_boxes"] = torch.tensor(oriented_boxes, dtype=torch.float32).reshape(-1, 5)
        if self.augmentation is not None:
            worker = get_worker_info()
            if worker is not None and self.augmentation_worker_seed != worker.seed:
                self.augmentation.transform.set_random_seed(worker.seed)
                self.augmentation_worker_seed = worker.seed
            polygons = [[np.asarray(segment, dtype=np.float32).reshape(-1, 2)
                         for segment in annotation["segmentation"]]
                        for annotation in self.annotations[info["id"]]]
            rgb, target = self.augmentation(rgb, target, polygons)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255)
        return tensor, target


def collate_detection(batch: list[tuple[Tensor, DetectionTarget]]) -> DetectionBatch:
    # Detection keeps variable image sizes and object counts rather than stacking targets.
    return [item[0] for item in batch], [item[1] for item in batch]


class EpochImageSampler(Sampler[int]):
    """Sample a reproducible subset without replacement, varying with the epoch."""

    def __init__(self, dataset_size: int, image_count: int, seed: int) -> None:
        self.dataset_size = dataset_size
        self.image_count = min(dataset_size, image_count)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.image_count

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.dataset_size, generator=generator)[:self.image_count].tolist())


class CardDataModule(pl.LightningDataModule):
    def __init__(
        self, train_root: Path, val_root: Path | None = None, batch_size: int = 2,
        num_workers: int = 4, val_fraction: float = 0.1, seed: int = 42,
        variant: DetectorVariant = DetectorVariant.BOX,
        test_root: Path | None = None,
        augment: bool = True,
        train_images_per_epoch: int | None = 200,
        val_images_per_epoch: int | None = 20,
        augmentation_config: AugmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.variant = variant
        self.test_root = test_root
        self.augment = augment
        self.augmentation_config = augmentation_config
        if train_images_per_epoch is not None and train_images_per_epoch < 1:
            raise ValueError("train_images_per_epoch must be positive or None")
        if val_images_per_epoch is not None and val_images_per_epoch < 1:
            raise ValueError("val_images_per_epoch must be positive or None")
        self.val_images_per_epoch = val_images_per_epoch
        self.train_images_per_epoch = train_images_per_epoch
        self.test_dataset = CocoCardDataset(test_root, variant) if test_root is not None else None

    def setup(self, stage: str | None = None) -> None:
        dataset = CocoCardDataset(self.train_root, self.variant)
        augmented_dataset = CocoCardDataset(self.train_root, self.variant,
                                           CardAugmentation(self.seed, self.augmentation_config) if self.augment else None)
        if self.val_root is not None:
            if self.val_root.resolve() == self.train_root.resolve():
                raise ValueError("Training and validation roots must differ")
            self.train_dataset = augmented_dataset
            self.val_dataset = CocoCardDataset(self.val_root, self.variant)
            train_indices = list(range(len(dataset)))
            val_indices = list(range(len(self.val_dataset)))
            val_source = self.val_dataset
        else:
            if len(dataset) < 2 or not 0 < self.val_fraction < 1:
                raise ValueError("Splitting requires at least two images and 0 < val_fraction < 1")
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(self.seed)).tolist()
            val_count = min(len(dataset) - 1, max(1, round(len(dataset) * self.val_fraction)))
            val_indices, train_indices = indices[:val_count], indices[val_count:]
            self.train_dataset = Subset(augmented_dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
            val_source = dataset
        if not len(self.train_dataset) or not len(self.val_dataset):
            raise ValueError("Training and validation datasets must both contain images")
        val_count = min(len(self.val_dataset), self.val_images_per_epoch or len(self.val_dataset))
        # Keep the same validation subset across epochs so checkpoint comparisons stay consistent.
        self.val_epoch_indices = sorted(torch.randperm(
            len(self.val_dataset), generator=torch.Generator().manual_seed(self.seed + 1),
        )[:val_count].tolist())
        self.split_manifest = {
            "seed": self.seed,
            "train_root": str(self.train_root.resolve()),
            "val_root": str((self.val_root or self.train_root).resolve()),
            "train_image_ids": [dataset.images[i]["id"] for i in train_indices],
            "val_image_ids": [val_source.images[i]["id"] for i in val_indices],
            "val_images_per_epoch": val_count,
            "val_epoch_image_ids": [val_source.images[val_indices[i]]["id"] for i in self.val_epoch_indices],
            "train_images_per_epoch": min(len(self.train_dataset), self.train_images_per_epoch)
            if self.train_images_per_epoch is not None else len(self.train_dataset),
        }

    def train_dataloader(self) -> DataLoader:
        sampler = EpochImageSampler(len(self.train_dataset),
                                    self.train_images_per_epoch or len(self.train_dataset), self.seed)
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=sampler,
            num_workers=self.num_workers, collate_fn=collate_detection,
            persistent_workers=self.num_workers > 0,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, sampler=self.val_epoch_indices,
            num_workers=self.num_workers, collate_fn=collate_detection,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Set test_root to evaluate a test dataset")
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers,
                          collate_fn=collate_detection)
