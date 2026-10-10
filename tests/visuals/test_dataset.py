import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from set_master.generation.synthetic_dataset import DatasetConfig, generate_coco_dataset


PALETTE = [
    (40, 80, 255), (50, 210, 70), (255, 110, 35), (230, 50, 210),
    (30, 220, 220), (210, 190, 35), (170, 70, 255), (70, 220, 150),
]


def test_generate_default_image_and_debug_annotations(output_dir: Path) -> None:
    """Generate one default scene and compare its mask with COCO annotations."""
    dataset_dir = output_dir / "dataset_example"
    shutil.rmtree(dataset_dir, ignore_errors=True)
    run_dir = generate_coco_dataset(
        DatasetConfig(image_count=1, output_root=output_dir, run_name=dataset_dir.name)
    )
    coco = json.loads((run_dir / "annotations.json").read_text(encoding="utf-8"))
    image_info = coco["images"][0]
    image = cv2.imread(str(run_dir / image_info["file_name"]))
    mask = cv2.imread(str(run_dir / image_info["mask_file"]), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert mask is not None
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()).issubset({0, 255})
    assert 255 in np.unique(mask)

    annotations = [
        annotation
        for annotation in coco["annotations"]
        if annotation["image_id"] == image_info["id"]
    ]
    overlay = image.copy()
    debug = image.copy()
    polygon_union = np.zeros_like(mask)

    for index, annotation in enumerate(annotations):
        polygon_mask = np.zeros_like(mask)
        for polygon in annotation["segmentation"]:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(polygon_mask, [points], 255)
            cv2.polylines(debug, [points], True, (0, 255, 255), 2)
        polygon_union[polygon_mask > 0] = 255

        ys, xs = np.nonzero(polygon_mask)
        assert xs.size > 0
        polygon_bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        assert annotation["bbox"] == polygon_bbox

        color = PALETTE[index % len(PALETTE)]
        overlay[polygon_mask > 0] = color
        contours, _ = cv2.findContours(polygon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, contours, -1, (255, 255, 0), 2)
        x, y, width, height = annotation["bbox"]
        cv2.rectangle(debug, (x, y), (x + width - 1, y + height - 1), (0, 0, 255), 2)
        cv2.putText(
            debug,
            f"annotation {annotation['id']}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    intersection = np.count_nonzero((mask > 0) & (polygon_union > 0))
    union = np.count_nonzero((mask > 0) | (polygon_union > 0))
    assert union > 0
    assert intersection / union >= 0.95

    debug = cv2.addWeighted(overlay, 0.45, debug, 0.55, 0)
    debug_path = run_dir / "debug.png"
    assert cv2.imwrite(str(debug_path), debug)
    assert (run_dir / image_info["file_name"]).is_file()
    assert (run_dir / image_info["mask_file"]).is_file()
    assert (run_dir / "annotations.json").is_file()
    assert debug_path.is_file()
    print(f"Synthetic dataset debug image: {debug_path}")
