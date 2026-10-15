"""Recover card-relative corner identities from saved synthetic render transforms."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from set_master.geometry import order_corners


def migrate(dataset: Path) -> None:
    path = dataset / "annotations.json"
    coco = json.loads(path.read_text())
    count = 0
    for annotation in coco["annotations"]:
        if annotation.get("corner_order") == "card_relative":
            continue
        render = annotation["render_transform"]
        points = np.float32(annotation["keypoints"]).reshape(4, 3)
        ordered = order_corners(points[:, :2])
        local = ordered.copy()
        if "scene_transform" in annotation:
            local = cv2.perspectiveTransform(
                local[None],
                np.linalg.inv(np.asarray(annotation["scene_transform"], dtype=np.float64)),
            )[0]
        rotation = cv2.getRotationMatrix2D((0, 0), render["rotation_degrees"], 1)[:, :2]
        shear = np.array(
            [
                [1, math.tan(math.radians(render.get("skew_x_degrees", 0)))],
                [math.tan(math.radians(render.get("skew_y_degrees", 0))), 1],
            ]
        )
        local = local @ np.linalg.inv(rotation @ shear).T
        edges = np.roll(local, -1, axis=0) - local
        # In sprite coordinates the top short edge points along positive x.
        horizontal = edges[:, 0] / np.linalg.norm(edges, axis=1)
        start = int(horizontal.argmax())
        if horizontal[start] < 0.8:
            raise ValueError(f"Ambiguous render orientation for card {annotation['id']} in {path}")
        ordered = np.roll(ordered, -start, axis=0)
        indices = np.linalg.norm(ordered[:, None] - points[None, :, :2], axis=2).argmin(axis=1)
        annotation["keypoints"] = points[indices].reshape(-1).tolist()
        annotation["corner_order"] = "card_relative"
        annotation["orientation_source"] = "recovered_render_transform"
        count += 1
    path.write_text(json.dumps(coco, indent=2) + "\n")
    print(f"Migrated {count} cards in {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    migrate(parser.parse_args().dataset)
