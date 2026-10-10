import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from set_master.geometry import geometry_annotation
from set_master.annotation_visualization import draw_annotation_geometry

from set_master.paths import (
    REAL_TESTSET_DIR,
    REFERENCE_ANNOTATIONS_PATH,
    REFERENCE_IMAGES_DIR,
)


def export_testset(output_dir: Path) -> None:
    labels = json.loads(REFERENCE_ANNOTATIONS_PATH.read_text())
    for directory in ("images", "masks", "instances", "debug"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    coco = {
        "info": {
            "description": "Reference SET card photos with reviewed card polygons",
            "annotation_scope": labels["scope"],
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "card", "supercategory": "card"}],
    }
    manifest = []
    annotation_id = 1
    for image_id, record in enumerate(labels["images"], start=1):
        source = REFERENCE_IMAGES_DIR / record["source"]
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(source)
        height, width = image.shape[:2]
        if (width, height) != (record["width"], record["height"]):
            raise ValueError(f"Source dimensions changed: {source}")
        stem = f"{image_id:06d}"
        image_file = f"images/{stem}.jpg"
        mask_file = f"masks/{stem}.png"
        instance_file = f"instances/{stem}.png"
        shutil.copyfile(source, output_dir / image_file)
        union = np.zeros((height, width), dtype=np.uint8)
        instances = np.zeros((height, width), dtype=np.uint16)
        colors = np.zeros_like(image)
        debug = image.copy()
        if "corners" not in record:
            raise ValueError(f"Missing explicit corner annotations in {source}")
        for instance_id, vertices in enumerate(record["polygons"], start=1):
            polygon = np.asarray(vertices, dtype=np.int32)
            if (
                polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or np.any(polygon < 0)
                or np.any(polygon[:, 0] >= width)
                or np.any(polygon[:, 1] >= height)
            ):
                raise ValueError(f"Invalid polygon {instance_id} in {source}")
            mask = np.zeros_like(union)
            cv2.fillPoly(mask, [polygon], 255)
            x, y, w, h = cv2.boundingRect(polygon)
            area = int(np.count_nonzero(mask))
            if not area:
                raise ValueError(f"Empty card mask in {source}")
            if np.any(instances[mask > 0]):
                raise ValueError(f"Overlapping card annotations in {source}")
            union[mask > 0] = 255
            instances[mask > 0] = instance_id
            color = tuple(int(value) for value in cv2.cvtColor(
                np.uint8([[[instance_id * 37 % 180, 180, 255]]]),
                cv2.COLOR_HSV2BGR,
            )[0, 0])
            colors[mask > 0] = color
            coco["annotations"].append({
                "id": annotation_id, "image_id": image_id, "category_id": 1,
                "segmentation": [polygon.reshape(-1).tolist()],
                "bbox": [x, y, w, h], "area": area, "iscrowd": 0,
                "instance_id": instance_id,
                **geometry_annotation(np.float32(record["corners"][instance_id - 1]), polygon.astype(np.float32)),
                "geometry_source": record["corner_annotation_method"],
            })
            annotation_id += 1
        debug[union > 0] = cv2.addWeighted(debug, 0.7, colors, 0.3, 0)[union > 0]
        image_annotations = [item for item in coco["annotations"] if item["image_id"] == image_id]
        debug = draw_annotation_geometry(debug, image_annotations)
        for filename, pixels in ((mask_file, union), (instance_file, instances),
                                 (f"debug/{stem}.jpg", debug)):
            if not cv2.imwrite(str(output_dir / filename), pixels):
                raise OSError(f"Could not write {output_dir / filename}")
        coco["images"].append({
            "id": image_id, "file_name": image_file, "width": width, "height": height,
            "mask_file": mask_file, "instance_mask_file": instance_file,
            "debug_file": f"debug/{stem}.jpg",
            "source_file_name": source.name,
            "original_source_file_name": record["original_source"],
        })
        manifest.append({
            "image_id": image_id, "file_name": image_file, "source_file_name": source.name,
            "original_source_file_name": record["original_source"],
            "scene_group": record["scene_group"], "derived_from": record.get("derived_from"),
            "annotation_method": record["annotation_method"],
            "card_count": len(record["polygons"]),
        })
    coco["categories"][0].update(keypoints=["top_left", "top_right", "bottom_right", "bottom_left"],
                                 skeleton=[[1, 2], [2, 3], [3, 4], [4, 1]])
    (output_dir / "annotations.json").write_text(json.dumps(coco, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(manifest)} images and {annotation_id - 1} cards to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REAL_TESTSET_DIR)
    args = parser.parse_args()
    export_testset(args.output_dir)
