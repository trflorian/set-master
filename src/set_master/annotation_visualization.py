from collections.abc import Iterable, Mapping

import cv2
import numpy as np
from numpy.typing import NDArray

from set_master.geometry import obb_corners


def draw_annotation_geometry(
    image: NDArray[np.uint8], annotations: Iterable[Mapping[str, object]], max_side: int = 1440,
) -> NDArray[np.uint8]:
    """Show mask boundaries, axis boxes, oriented boxes, and ordered sharp corners."""
    height, width = image.shape[:2]
    scale = min(1.0, (max_side - 32) / max(height, width))
    preview = cv2.resize(image, (round(width * scale), round(height * scale)))
    for index, annotation in enumerate(annotations, start=1):
        if "segmentation" in annotation:
            for polygon in np.asarray(annotation["segmentation"], dtype=np.float32):
                points = np.rint(polygon.reshape(-1, 2) * scale).astype(np.int32)
                cv2.polylines(preview, [points], True, (255, 255, 0), 1)
        x, y, w, h = np.asarray(annotation["bbox"], dtype=np.float32)
        first = (round(float(x * scale)), round(float(y * scale)))
        last = (round(float((x + w - 1) * scale)), round(float((y + h - 1) * scale)))
        cv2.rectangle(preview, first, last, (0, 180, 0), 1)
        cv2.putText(preview, f"#{index}", (first[0], max(15, first[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)
        if "oriented_bbox" in annotation:
            box = np.asarray(annotation["oriented_bbox"], dtype=np.float32).tolist()
            points = np.rint(obb_corners(box) * scale).astype(np.int32)
            cv2.polylines(preview, [points], True, (255, 100, 30), 2)
        if "keypoints" in annotation:
            corners = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(4, 3)[:, :2]
            corners = np.rint(corners * scale).astype(np.int32)
            cv2.polylines(preview, [corners], True, (0, 165, 255), 2)
            for corner_index, (cx, cy) in enumerate(corners, start=1):
                cv2.circle(preview, (int(cx), int(cy)), 4, (0, 165, 255), -1)
                cv2.putText(preview, str(corner_index), (int(cx) + 5, int(cy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    preview = cv2.copyMakeBorder(preview, 0, 32, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    position = 10
    for label, color in [("BBox", (0, 180, 0)), ("OBB", (255, 100, 30)),
                         ("Corners 1-4", (0, 165, 255)), ("Mask", (255, 255, 0))]:
        cv2.putText(preview, label, (position, preview.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        position += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 22
    return preview
