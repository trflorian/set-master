import argparse
import json
import math
from pathlib import Path

import cv2
import torch

from set_master.training.model import CardDetector, DEFAULT_REPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict card boxes and save crops for stage two")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/card_predictions"))
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("score-threshold must be between 0 and 1")
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read {args.image}")
    model = CardDetector.load_from_checkpoint(
        args.checkpoint, map_location="cpu", weights_only=True,
        repo_path=str(args.dinov3_repo.expanduser().resolve()), initialize_pretrained=False,
    ).eval().to(args.device)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(args.device)
    with torch.inference_mode():
        prediction = model([tensor])[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    debug = image.copy()
    detections = []
    for box, score in zip(prediction["boxes"].cpu().tolist(), prediction["scores"].cpu().tolist()):
        if score < args.score_threshold:
            continue
        x1, y1, x2, y2 = box
        crop_box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
        left, top, right, bottom = crop_box
        crop_name = f"card_{len(detections):03d}.png"
        if not cv2.imwrite(str(crops_dir / crop_name), image[top:bottom, left:right]):
            raise OSError(f"Could not save crop {crop_name}")
        detections.append({"bbox_xyxy": box, "score": score, "crop_box_xyxy": crop_box, "crop": f"crops/{crop_name}"})
        cv2.rectangle(debug, (left, top), (right - 1, bottom - 1), (0, 200, 0), 2)
        cv2.putText(debug, f"card {score:.2f}", (left, max(16, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    (args.output_dir / "predictions.json").write_text(json.dumps({"image": str(args.image), "detections": detections}, indent=2) + "\n")
    if not cv2.imwrite(str(args.output_dir / "debug.jpg"), debug):
        raise OSError("Could not save prediction debug image")
    print(f"Saved {len(detections)} card boxes and crops to {args.output_dir}")


if __name__ == "__main__":
    main()
