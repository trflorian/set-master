import argparse
import json
import math
from pathlib import Path

import cv2
import torch
import numpy as np

from set_master.training.model import DEFAULT_REPO
from set_master.training.models.factory import load_detector
from set_master.geometry import obb_corners, rectify_card
from set_master.paths import RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict card boxes and save crops for stage two")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "card_predictions")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("score-threshold must be between 0 and 1")
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read {args.image}")
    model = load_detector(args.checkpoint, args.dinov3_repo.expanduser().resolve()).to(args.device)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(args.device)
    with torch.inference_mode():
        prediction = model([tensor])[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    debug = image.copy()
    detections = []
    for index, (box, score) in enumerate(zip(prediction["boxes"].cpu().tolist(), prediction["scores"].cpu().tolist())):
        if score < args.score_threshold:
            continue
        x1, y1, x2, y2 = box
        crop_box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
        left, top, right, bottom = crop_box
        crop_name = f"card_{len(detections):03d}.png"
        points = None
        if "oriented_boxes" in prediction:
            points = obb_corners(prediction["oriented_boxes"][index].cpu().tolist())
        elif "keypoints" in prediction:
            points = np.ascontiguousarray(prediction["keypoints"][index, :, :2].cpu().numpy())
        crop = image[top:bottom, left:right]
        if points is not None and cv2.isContourConvex(points) and abs(cv2.contourArea(points)) >= 1:
            crop = rectify_card(image, points, card_relative='keypoints' in prediction)
        if not cv2.imwrite(str(crops_dir / crop_name), crop):
            raise OSError(f"Could not save crop {crop_name}")
        detections.append({"bbox_xyxy": box, "score": score, "crop_box_xyxy": crop_box, "crop": f"crops/{crop_name}"})
        if points is not None:
            detections[-1]["corners"] = points.tolist()
            cv2.polylines(debug, [points.round().astype("int32")], True, (0, 200, 0), 2)
            if 'keypoints' in prediction:
                axis = np.rint([(points[0] + points[1]) / 2,
                                (points[2] + points[3]) / 2]).astype(np.int32)
                cv2.line(debug, tuple(axis[0]), tuple(axis[1]), (255, 0, 255), 3, cv2.LINE_AA)
        cv2.rectangle(debug, (left, top), (right - 1, bottom - 1), (0, 200, 0), 2)
        cv2.putText(debug, f"card {score:.2f}", (left, max(16, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    (args.output_dir / "predictions.json").write_text(json.dumps({"image": str(args.image), "detections": detections}, indent=2) + "\n")
    if not cv2.imwrite(str(args.output_dir / "debug.jpg"), debug):
        raise OSError("Could not save prediction debug image")
    print(f"Saved {len(detections)} card boxes and crops to {args.output_dir}")


if __name__ == "__main__":
    main()
