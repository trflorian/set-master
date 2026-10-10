import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from time import perf_counter

import cv2
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from set_master.training.data import CocoCardDataset, collate_detection
from set_master.training.model import CardDetector
from set_master.training.config import DetectorVariant
from set_master.training.geometry_metrics import GeometryMetrics
from set_master.geometry import obb_corners, rectify_card


class ImageResult(TypedDict):
    image_id: int
    file_name: str
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    images: list[ImageResult]


def matched_count(boxes: Tensor, scores: Tensor, targets: Tensor, iou_threshold: float) -> int:
    """Match predictions in confidence order, using each ground-truth card once."""
    if not len(boxes) or not len(targets):
        return 0
    overlaps = box_iou(boxes[scores.argsort(descending=True)], targets)
    available = torch.ones(len(targets), dtype=torch.bool)
    count = 0
    for row in overlaps:
        values = row.masked_fill(~available, -1)
        value, index = values.max(dim=0)
        if value >= iou_threshold:
            available[index] = False
            count += 1
    return count


def prediction_overlay(
    image: Tensor, target: dict[str, Tensor], prediction: dict[str, Tensor],
    score_threshold: float, max_side: int = 1440,
) -> cv2.typing.MatLike:
    rgb = image.cpu().mul(255).byte().permute(1, 2, 0).numpy()
    preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = preview.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    preview = cv2.resize(preview, (round(width * scale), round(height * scale)))
    for box in target["boxes"].cpu():
        x1, y1, x2, y2 = (box * scale).round().int().tolist()
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 180, 0), 2)
    if "oriented_boxes" in prediction and "oriented_boxes" in target:
        for box in target["oriented_boxes"].cpu().tolist():
            cv2.polylines(preview, [np.rint(obb_corners(box) * scale).astype(np.int32)], True, (0, 180, 0), 2)
    elif "keypoints" in prediction and "keypoints" in target:
        for points in target["keypoints"][..., :2].cpu().numpy():
            cv2.polylines(preview, [np.rint(points * scale).astype(np.int32)], True, (0, 180, 0), 2)
    for index, (box, score) in enumerate(zip(prediction["boxes"].cpu(), prediction["scores"].cpu())):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = (box * scale).round().int().tolist()
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 220), 2)
        points = None
        if "oriented_boxes" in prediction:
            points = obb_corners(prediction["oriented_boxes"][index].cpu().tolist())
        elif "keypoints" in prediction:
            points = prediction["keypoints"][index, :, :2].cpu().numpy()
        if points is not None:
            display_points = np.rint(points * scale).astype(np.int32)
            cv2.polylines(preview, [display_points], True, (0, 0, 220), 2)
            for corner_index, (cx, cy) in enumerate(display_points):
                cv2.circle(preview, (int(cx), int(cy)), 4, (0, 0, 220), -1)
                cv2.putText(preview, str(corner_index + 1), (int(cx) + 4, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
        cv2.putText(preview, f"{score:.2f}", (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    return preview


def evaluate_detector(
    model: CardDetector, dataset: CocoCardDataset, output_dir: Path,
    image_ids: list[int] | None = None, overlay_image_ids: tuple[int, ...] = (),
    batch_size: int = 1, num_workers: int = 0,
    score_threshold: float = 0.5, iou_threshold: float = 0.5,
) -> EvaluationResult:
    if model.training:
        raise ValueError("Call model.eval() before evaluation")
    if not 0 <= score_threshold <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("Require confidence in [0, 1] and IoU in (0, 1]")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Require batch_size > 0 and num_workers >= 0")
    available = {info["id"] for info in dataset.images}
    selected = available if image_ids is None else set(image_ids)
    if not selected or selected - available:
        raise ValueError("Evaluation image IDs must be nonempty and present in the dataset")
    if set(overlay_image_ids) - selected:
        raise ValueError("Overlay image IDs must belong to the evaluation selection")
    subset = Subset(dataset, [i for i, info in enumerate(dataset.images) if info["id"] in selected])
    loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=collate_detection, shuffle=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", compute_on_cpu=True)
    geometry_metric = GeometryMetrics(model.variant) if model.variant != DetectorVariant.BOX else None
    info_by_id = {info["id"]: info for info in dataset.images}
    rows: list[ImageResult] = []
    detections = []
    inference_seconds = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            started = perf_counter()
            predictions = model([image.to(model.device) for image in images])
            predictions = [{key: value.cpu() for key, value in pred.items()} for pred in predictions]
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            inference_seconds += perf_counter() - started
            # AP uses all detector outputs; the display threshold only affects counts and overlays.
            metric.update(predictions, targets)
            if geometry_metric is not None:
                geometry_metric.update(predictions, targets)
            for image, target, prediction in zip(images, targets, predictions):
                image_id = int(target["image_id"].item())
                keep = prediction["scores"] >= score_threshold
                boxes, scores = prediction["boxes"][keep], prediction["scores"][keep]
                tp = matched_count(boxes, scores, target["boxes"], iou_threshold)
                rows.append({
                    "image_id": image_id, "file_name": info_by_id[image_id]["file_name"],
                    "ground_truth": len(target["boxes"]), "predictions": len(boxes),
                    "true_positives": tp, "false_positives": len(boxes) - tp,
                    "false_negatives": len(target["boxes"]) - tp,
                })
                for index, (box, score) in enumerate(zip(prediction["boxes"].tolist(), prediction["scores"].tolist())):
                    x1, y1, x2, y2 = box
                    detection = {"image_id": image_id, "category_id": 1,
                                 "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score}
                    if "oriented_boxes" in prediction:
                        detection["oriented_bbox"] = prediction["oriented_boxes"][index].tolist()
                    if "keypoints" in prediction:
                        detection["keypoints"] = prediction["keypoints"][index].reshape(-1).tolist()
                    detections.append(detection)
                if image_id in overlay_image_ids:
                    preview = prediction_overlay(image, target, prediction, score_threshold)
                    filename = output_dir / f"overlay_{image_id:06d}.jpg"
                    if not cv2.imwrite(str(filename), preview):
                        raise OSError(f"Could not save {filename}")
                    rgb = image.mul(255).byte().permute(1, 2, 0).numpy()
                    source = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    crop_dir = output_dir / "crops" / f"{image_id:06d}"
                    for index in torch.where(keep)[0].tolist():
                        if "oriented_boxes" in prediction:
                            points = obb_corners(prediction["oriented_boxes"][index].tolist())
                        elif "keypoints" in prediction:
                            points = prediction["keypoints"][index, :, :2].numpy()
                        else:
                            continue
                        points = np.ascontiguousarray(points, dtype=np.float32)
                        if not cv2.isContourConvex(np.float32(points)) or abs(cv2.contourArea(np.float32(points))) < 1:
                            continue
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(crop_dir / f"card_{index:03d}.png"), rectify_card(source, points)):
                            raise OSError(f"Could not save crop in {crop_dir}")
            print(f"Evaluated {len(rows)}/{len(subset)} images", end="\r")
    print()
    raw = metric.compute()
    metrics = {name: float(raw[name].item()) for name in ("map", "map_50", "map_75", "mar_100")}
    metrics["inference_seconds"] = inference_seconds
    metrics["images_per_second"] = len(rows) / inference_seconds if inference_seconds else 0.0
    if geometry_metric is not None:
        metrics.update(geometry_metric.compute())
    tp = sum(row["true_positives"] for row in rows)
    fp = sum(row["false_positives"] for row in rows)
    fn = sum(row["false_negatives"] for row in rows)
    metrics.update(precision=tp / (tp + fp) if tp + fp else 0.0,
                   recall=tp / (tp + fn) if tp + fn else 0.0,
                   f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    report = {
        "dataset": str(dataset.root.resolve()), "image_ids": sorted(selected),
        "variant": model.variant.value,
        "score_threshold": score_threshold, "iou_threshold": iou_threshold,
        "metrics": metrics, "images": rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "predictions.json").write_text(json.dumps(detections, indent=2) + "\n")
    return EvaluationResult(metrics=metrics, images=rows)
