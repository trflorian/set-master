import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from set_master.training.augmentation import AugmentationConfig, CardAugmentation
from set_master.training.data import CocoCardDataset


@dataclass(frozen=True)
class BenchmarkResult:
    scenes: np.ndarray
    crops: np.ndarray
    milliseconds_per_image: float
    original_cards: int
    retained_cards: int
    image_ids: list[int]
    source_root: Path

    def save(self, directory: Path, config: AugmentationConfig, seed: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, image in [('scenes', self.scenes), ('crops', self.crops)]:
            if not cv2.imwrite(str(directory / f'{name}.png'), image):
                raise OSError(f'Could not save {name} preview')
        (directory / 'benchmark.json').write_text(json.dumps({
            'seed': seed, 'image_ids': self.image_ids, 'config': asdict(config),
            'dataset': str(self.source_root.resolve()),
            'milliseconds_per_image': self.milliseconds_per_image,
            'original_cards': self.original_cards, 'retained_cards': self.retained_cards,
        }, indent=2) + '\n')


def _tile(image: np.ndarray, label: str, boxes: np.ndarray | None = None) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(480 / width, 270 / height)
    resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (round(width * scale), round(height * scale)))
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = np.rint(box * scale).astype(int)
            cv2.rectangle(resized, (x1, y1), (x2 - 1, y2 - 1), (0, 190, 0), 1)
    canvas = np.full((300, 480, 3), 225, dtype=np.uint8)
    canvas[30:30 + resized.shape[0], :resized.shape[1]] = resized
    cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    return canvas


def render_augmentation_benchmark(
    root: Path, image_ids: list[int], config: AugmentationConfig, seed: int = 42,
    show_boxes: bool = False,
) -> BenchmarkResult:
    """Compare explicit source images with one reproducible draw, including matched card crops."""
    if not image_ids or len(image_ids) > 6:
        raise ValueError('Choose between one and six image IDs per preview')
    dataset = CocoCardDataset(root)
    index_by_id = {info['id']: index for index, info in enumerate(dataset.images)}
    augmentation = CardAugmentation(seed, config)
    scene_tiles, crop_tiles = [], []
    elapsed, original_cards, retained_cards = 0.0, 0, 0
    for image_id in image_ids:
        original, target = dataset[index_by_id[image_id]]
        image = original.mul(255).byte().permute(1, 2, 0).numpy()
        target['source_indices'] = torch.arange(len(target['boxes']))
        polygons = [[np.asarray(segment, dtype=np.float32).reshape(-1, 2)
                     for segment in annotation['segmentation']]
                    for annotation in dataset.annotations[image_id]]
        start = perf_counter()
        augmented, after = augmentation(image.copy(), target, polygons)
        elapsed += perf_counter() - start
        original_cards += len(target['boxes'])
        retained_cards += len(after['boxes'])
        scene_tiles.extend([
            _tile(image, f'{image_id} original', target['boxes'].numpy() if show_boxes else None),
            _tile(augmented, f'{image_id} augmented ({len(after["boxes"])}/{len(target["boxes"])} cards)',
                  after['boxes'].numpy() if show_boxes else None),
        ])
        if len(after['boxes']):
            candidates = after['source_indices']
            original_areas = target['area'][candidates]
            selected = int(original_areas.argmax())
            source_index = int(candidates[selected])
            pair = []
            for array, box, label in [(image, target['boxes'][source_index], 'original crop'),
                                      (augmented, after['boxes'][selected], 'augmented crop')]:
                x1, y1, x2, y2 = box.numpy()
                margin = max(x2 - x1, y2 - y1) * 0.18
                left, top = max(0, int(x1 - margin)), max(0, int(y1 - margin))
                right, bottom = min(array.shape[1], int(x2 + margin)), min(array.shape[0], int(y2 + margin))
                pair.append(_tile(array[top:bottom, left:right], f'{image_id} {label}'))
            crop_tiles.extend(pair)
        else:
            crop_tiles.extend([_tile(image, f'{image_id} no retained cards')] * 2)
    # Two before/after pairs per row; preserve the final empty slot for odd sample counts.
    if len(scene_tiles) % 4:
        empty = np.full_like(scene_tiles[0], 225)
        scene_tiles.extend([empty] * 2)
        crop_tiles.extend([empty] * 2)
    scenes = np.concatenate([np.concatenate(scene_tiles[i:i + 4], axis=1)
                             for i in range(0, len(scene_tiles), 4)], axis=0)
    crops = np.concatenate([np.concatenate(crop_tiles[i:i + 4], axis=1)
                            for i in range(0, len(crop_tiles), 4)], axis=0)
    return BenchmarkResult(scenes, crops, elapsed / len(image_ids) * 1000,
                           original_cards, retained_cards, image_ids, root)
