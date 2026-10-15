import argparse
import json
from pathlib import Path

import numpy as np
import cv2

from set_master.geometry import geometry_annotation, polygon_corners
from set_master.annotation_visualization import draw_annotation_geometry


def add_geometry(dataset: Path) -> None:
    path = dataset / "annotations.json"
    coco = json.loads(path.read_text())
    count = 0
    for annotation in coco["annotations"]:
        if (
            "keypoints" in annotation
            and "oriented_bbox" in annotation
            and annotation.get("geometry_source") != "polygon_edge_fit"
        ):
            continue
        segmentation = annotation["segmentation"]
        if not isinstance(segmentation, list) or len(segmentation) != 1:
            raise ValueError("Geometry fitting requires one polygon per card")
        polygon = np.float32(segmentation[0]).reshape(-1, 2)
        corners = polygon_corners(polygon)
        card_relative = annotation.get("corner_order") == "card_relative"
        if card_relative:
            old = np.float32(annotation["keypoints"]).reshape(4, 3)[:, :2]
            shifts = [np.roll(corners, -shift, axis=0) for shift in range(4)]
            corners = min(shifts, key=lambda points: float(np.linalg.norm(points - old, axis=1).sum()))
        annotation.update(geometry_annotation(corners, polygon, card_relative=card_relative))
        annotation["geometry_source"] = "polygon_edge_fit"
        count += 1
    for category in coco["categories"]:
        if category["name"] == "card":
            category.update(
                keypoints=["top_left", "top_right", "bottom_right", "bottom_left"],
                skeleton=[[1, 2], [2, 3], [3, 4], [4, 1]],
            )
    (dataset / "debug").mkdir(exist_ok=True)
    for info in coco["images"]:
        image = cv2.imread(str(dataset / info["file_name"]))
        if image is None:
            raise FileNotFoundError(dataset / info["file_name"])
        annotations = [item for item in coco["annotations"] if item["image_id"] == info["id"]]
        filename = f"debug/{info['id']:06d}.jpg"
        if not cv2.imwrite(str(dataset / filename), draw_annotation_geometry(image, annotations)):
            raise OSError(f"Could not save {filename}")
        info["debug_file"] = filename
    path.write_text(json.dumps(coco, indent=2) + "\n")
    print(f"Added estimated geometry to {count} cards in {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    add_geometry(args.dataset)
