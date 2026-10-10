import numpy as np
from torch import Tensor

from set_master.geometry import obb_corners, polygon_iou
from set_master.training.config import DetectorVariant


class GeometryMetrics:
    """Single-class, 101-point AP for rotated rectangles or four-corner polygons."""

    def __init__(self, variant: DetectorVariant) -> None:
        self.variant = variant
        self.records: list[tuple[np.ndarray, np.ndarray]] = []
        self.corner_errors: list[float] = []
        self.matched_ious: list[float] = []

    def update(self, predictions: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]) -> None:
        for prediction, target in zip(predictions, targets):
            if self.variant == DetectorVariant.ORIENTED:
                predicted = np.float32([obb_corners(box) for box in prediction["oriented_boxes"].detach().cpu().tolist()]).reshape(-1, 4, 2)
                expected = np.float32([obb_corners(box) for box in target["oriented_boxes"].detach().cpu().tolist()]).reshape(-1, 4, 2)
            else:
                predicted = prediction["keypoints"][..., :2].detach().cpu().numpy()
                expected = target["keypoints"][..., :2].detach().cpu().numpy()
            scores = prediction["scores"].detach().cpu().numpy()
            order = np.argsort(-scores, kind="stable")[:100]
            predicted, scores = predicted[order], scores[order]
            overlaps = np.zeros((len(predicted), len(expected)), dtype=np.float32)
            for row, points in enumerate(predicted):
                for column, ground_truth in enumerate(expected):
                    overlaps[row, column] = polygon_iou(points, ground_truth)
            self.records.append((scores, overlaps))
            available = set(range(len(expected)))
            for row in range(len(predicted)):
                if not available:
                    break
                column = max(available, key=lambda index: overlaps[row, index])
                if overlaps[row, column] >= 0.5:
                    available.remove(column)
                    self.matched_ious.append(float(overlaps[row, column]))
                    if self.variant in (DetectorVariant.CORNERS, DetectorVariant.QUADRILATERAL):
                        diagonal = np.linalg.norm(expected[column].max(axis=0) - expected[column].min(axis=0))
                        error = np.linalg.norm(predicted[row] - expected[column], axis=1).mean()
                        self.corner_errors.append(float(error / max(diagonal, 1)))

    def compute(self) -> dict[str, float]:
        ground_truth_count = sum(overlaps.shape[1] for _, overlaps in self.records)
        aps = []
        for threshold in np.linspace(0.5, 0.95, 10):
            detections = []
            for scores, overlaps in self.records:
                available = set(range(overlaps.shape[1]))
                for row, score in enumerate(scores):
                    match = max(available, key=lambda index: overlaps[row, index]) if available else None
                    matched = match is not None and overlaps[row, match] >= threshold
                    if matched:
                        available.remove(match)
                    detections.append((float(score), int(matched)))
            detections.sort(key=lambda item: -item[0])
            true_positive = np.cumsum([matched for _, matched in detections])
            recall = true_positive / max(ground_truth_count, 1)
            precision = true_positive / np.arange(1, len(detections) + 1)
            interpolated = [float(precision[recall >= level].max()) if np.any(recall >= level) else 0.0
                            for level in np.linspace(0, 1, 101)]
            aps.append(float(np.mean(interpolated)) if ground_truth_count else -1.0)
        result = {"geometry_map": float(np.mean(aps)), "geometry_map_50": aps[0],
                  "geometry_map_75": aps[5], "matched_polygon_iou": float(np.mean(self.matched_ious)) if self.matched_ious else -1.0}
        if self.variant in (DetectorVariant.CORNERS, DetectorVariant.QUADRILATERAL):
            result["corner_error_normalized"] = float(np.mean(self.corner_errors)) if self.corner_errors else -1.0
        return result

    def reset(self) -> None:
        self.records.clear()
        self.corner_errors.clear()
        self.matched_ious.clear()
