import numpy as np

from set_master.geometry import geometry_annotation, polygon_corners, polygon_iou, rectify_card


def test_corners_follow_long_edges_despite_clustered_contour_defects() -> None:
    polygon = np.float32([
        [508, 865], [508, 875], [498, 890], [495, 925], [500, 1280],
        [492, 1285], [500, 1290], [505, 1488], [510, 1508], [522, 1520],
        [538, 1525], [1472, 1520], [1492, 1505], [1498, 1490], [1492, 958],
        [1508, 958], [1492, 952], [1490, 880], [1478, 862], [1465, 858],
        [922, 860], [910, 858], [908, 850], [905, 862], [890, 862],
        [888, 848], [880, 842], [875, 862], [590, 865], [510, 870],
    ])
    expected = np.float32([[493, 867], [1490, 858], [1495, 1520], [506, 1525]])
    np.testing.assert_allclose(polygon_corners(polygon), expected, atol=10)


def test_polygon_iou_accepts_xy_slice_of_keypoints_with_visibility() -> None:
    keypoints = np.float32([[0, 0, 2], [100, 0, 2], [100, 150, 2], [0, 150, 2]])
    assert polygon_iou(keypoints[:, :2], keypoints[:, :2]) == 1.0


def test_card_relative_geometry_preserves_orientation_under_foreshortening() -> None:
    corners = np.float32([[0, 0], [79, 0], [79, 39], [0, 39]])
    # The semantic short edge projects longer than the long edge.
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:20] = [255, 0, 0]
    image[20:] = [0, 255, 0]
    annotation = geometry_annotation(corners, corners, card_relative=True)
    assert annotation['corner_order'] == 'card_relative'
    crop = rectify_card(image, corners, card_relative=True)
    np.testing.assert_array_equal(crop[20, 128], [255, 0, 0])
    np.testing.assert_array_equal(crop[360, 128], [0, 255, 0])
    turned = rectify_card(image, np.roll(corners, -2, axis=0), card_relative=True)
    np.testing.assert_array_equal(turned[20, 128], [0, 255, 0])
