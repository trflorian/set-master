import cv2
import numpy as np
from numpy.typing import NDArray


def order_corners(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Clockwise in image coordinates, starting at the smallest x+y corner."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    return np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)


def polygon_corners(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Estimate virtual sharp corners by fitting the straight portions of card edges."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for fraction in np.linspace(0.01, 0.08, 30):
        approximation = cv2.approxPolyDP(hull, float(fraction * perimeter), True)
        if len(approximation) == 4:
            quad = order_corners(approximation.reshape(4, 2))
            break
    if quad is None:
        raise ValueError("Could not estimate four card edges; annotate corners explicitly")
    # Contour simplification leaves long edges sparse and small defects dense.
    # Uniform sampling keeps those defects from dominating the line fit.
    sampled = []
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / 2)))
        sampled.append(np.linspace(start, end, count, endpoint=False, dtype=np.float32))
    points = np.concatenate(sampled)
    lines = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        direction = end - start
        length = np.linalg.norm(direction)
        unit = direction / length
        projection = (points - start) @ unit
        distance = np.abs((points[:, 0] - start[0]) * unit[1] - (points[:, 1] - start[1]) * unit[0])
        selected = points[(projection > length * 0.15) & (projection < length * 0.85)
                          & (distance < max(2.0, length * 0.04))]
        coverage = np.ptp((selected - start) @ unit) if len(selected) else 0
        if len(selected) >= 2 and coverage > length * 0.5:
            vx, vy, x, y = cv2.fitLine(selected, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4)
            lines.append((np.array([x, y]), np.array([vx, vy])))
        else:
            lines.append((start, unit))
    corners = []
    for index in range(4):
        p, u = lines[index - 1]
        q, v = lines[index]
        matrix = np.column_stack((u, -v))
        if abs(np.linalg.det(matrix)) < 1e-5:
            raise ValueError("Card edges are parallel; annotate corners explicitly")
        distance = np.linalg.solve(matrix, q - p)[0]
        corners.append(p + u * distance)
    return order_corners(np.asarray(corners, dtype=np.float32))


def oriented_box(points: NDArray[np.float32]) -> list[float]:
    """Return [cx, cy, short width, long height, clockwise angle in degrees]."""
    (cx, cy), (width, height), angle = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    if width > height:
        width, height = height, width
        angle += 90
    angle = (angle + 90) % 180 - 90
    return [float(cx), float(cy), float(width), float(height), float(angle)]


def obb_corners(box: list[float]) -> NDArray[np.float32]:
    cx, cy, width, height, angle = box
    return order_corners(cv2.boxPoints(((cx, cy), (width, height), angle)))


def polygon_iou(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    first = np.ascontiguousarray(first, dtype=np.float32).reshape(-1, 2)
    second = np.ascontiguousarray(second, dtype=np.float32).reshape(-1, 2)
    if not cv2.isContourConvex(first) or not cv2.isContourConvex(second):
        return 0.0
    area1, area2 = abs(cv2.contourArea(first)), abs(cv2.contourArea(second))
    if area1 <= 0 or area2 <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


def rectify_card(
    image: NDArray[np.uint8], corners: NDArray[np.float32], size: tuple[int, int] = (256, 384),
    *, card_relative: bool = False,
) -> NDArray[np.uint8]:
    source = np.asarray(corners, dtype=np.float32).reshape(4, 2) if card_relative else order_corners(corners)
    if not cv2.isContourConvex(source) or abs(cv2.contourArea(source)) < 1:
        raise ValueError("Card corners must form a nondegenerate convex quadrilateral")
    # Align the shorter edge with crop width; this does not infer printed orientation.
    if not card_relative and np.linalg.norm(source[1] - source[0]) > np.linalg.norm(source[2] - source[1]):
        source = np.roll(source, -1, axis=0)
    width, height = size
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, transform, size)


def geometry_annotation(
    corners: NDArray[np.float32], polygon: NDArray[np.float32],
    *, card_relative: bool = False,
) -> dict[str, list[float] | int | str]:
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2) if card_relative else order_corners(corners)
    return {
        "keypoints": [value for x, y in corners for value in (float(x), float(y), 2.0)],
        "num_keypoints": 4,
        "oriented_bbox": oriented_box(polygon),
        "corner_order": "card_relative" if card_relative else "image_relative",
    }
