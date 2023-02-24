import sys

import cv2
import os
import matplotlib.pyplot as plt

import numpy as np

from inspyrenet_inference import mask_image

img_dir = 'images_500x500'
img_dir_hr = 'images'
mask_dir = 'results'

# img_fn = 'PXL_20230205_150355105'
#img_fn = 'PXL_20230205_150409042'
#img_fn = 'PXL_20230205_150413474'
img_fn = 'PXL_20230205_150355105'

input_img_fn = f'{img_dir}/{img_fn}.jpg'
# input_img_fn = f'{img_dir_hr}/{img_fn}.jpg'
input_img = cv2.imread(input_img_fn)

# load High-Res image and determine scale factors
hr_image = cv2.imread(f'{img_dir_hr}/{img_fn}.jpg')
# hr_image = input_img

scale_x = hr_image.shape[0] / input_img.shape[0]
scale_y = hr_image.shape[1] / input_img.shape[1]

use_result = False

if use_result:
    mask = cv2.imread(f'{mask_dir}/{img_dir}/{img_fn}.png')
else:
    mask = mask_image(input_img_fn)

    mask = np.array(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

# mask = cv2.erode(mask, np.ones((3, 3)))
# mask = cv2.dilate(mask, np.ones((5, 5)))
mask = cv2.erode(mask, np.ones((3, 3)))

edges = cv2.Canny(mask, 0, 250)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_cont = mask.copy()
# cv2.drawContours(img_cont, contours, -1, [255, 255, 0], thickness=5)

all_cards = []

for cont in contours:
    # Approximate the contour
    epsilon = 0.05 * cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, epsilon, True)

    area = cv2.contourArea(approx)
    cv2.drawContours(img_cont, [approx], -1, [255, 255, 0], thickness=2)

    if len(approx) == 4 and area >= 500:
        all_cards.append(approx)

print(f'Found {len(all_cards)} Cards!')

if len(all_cards) < 12:
    print('Did not find all cards!')
    cards = all_cards
elif len(all_cards) > 12:
    print('Found more than 12 cards. Deleting outlier(s)...')

    print(all_cards[0].shape)

    # calculate the average x and y coordinates of all contours
    average_x = np.mean([c[:, 0, 0] for c in all_cards], axis=0)
    average_y = np.mean([c[:, 0, 1] for c in all_cards], axis=0)
    average = np.stack((average_x, average_y), axis=1)

    distances = []
    for card in all_cards:
        mean_x = np.mean(card[:, 0, 0])
        mean_y = np.mean(card[:, 0, 1])
        mean = np.stack((mean_x, mean_y), axis=0)
        distance = np.linalg.norm(mean - average)
        # print(distance)
        distances.append(distance)

    sorted_indices = np.argsort(distances)[:12]
    cards = [all_cards[i] for i in sorted_indices]
else:
    cards = all_cards

cv2.drawContours(input_img, cards, -1, (0, 255, 255), 2)
# cv2.drawContours(input_img, deleted_cards, -1, (0, 0, 255), 2)


card_images = []


def sort_points(points):
    # sort the points based on their x-coordinates
    points = points[np.argsort(points[:, 0]), :]

    # split the points into left and right halves
    left_half = points[:len(points) // 2, :]
    right_half = points[len(points) // 2:, :]

    # sort the left half based on their y-coordinates (top to bottom)
    left_half = left_half[np.argsort(left_half[:, 1]), :]

    # sort the right half based on their y-coordinates (bottom to top)
    right_half = right_half[np.argsort(-right_half[:, 1]), :]

    # return the concatenated and sorted points
    return np.concatenate((left_half, right_half), axis=0)


card_border = 0
card_width = 200
card_height = 350

for card in cards:
    # convert to correct format for wrap transform perspective
    card = card.reshape(-1, 2)
    card = card.astype(np.float32)

    # scale cards to OG image size
    card[:, 1] *= scale_x
    card[:, 0] *= scale_y

    card = sort_points(card)

    # card = sorted(card, key=lambda c: c[0])

    # Get the bounding rectangle of the card
    x, y, wbb, hbb = cv2.boundingRect(card)

    w = card_height + card_border * 2
    h = card_width + card_border * 2

    # print(f'{wbb=}, {hbb=}')
    if wbb > hbb:
        # print('swapping!')
        wt = w
        w = h
        h = wt

    target_box = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(card, target_box)

    # Apply the perspective transform to the original image
    flat_image = cv2.warpPerspective(hr_image, M, (w, h))

    # cimg = cv2.Canny(flat_image, 0, 250)
    # Setup SimpleBlobDetector parameters.
    # params = cv2.SimpleBlobDetector_Params()
    #
    # # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200
    #
    # # Filter by Area.
    # params.filterByArea = True
    # params.minArea = 1500
    #
    # # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1
    #
    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    #
    # # Set up the detector with default parameters.
    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(flat_image)
    #
    # im_with_keypoints = cv2.drawKeypoints(flat_image, keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    card_images.append(flat_image)
    # cv2.imshow(f'unwrapped_{card}', flat_image)

card_images = [cimg[card_border:card_height - card_border, card_border:card_width - card_border] for cimg in
               card_images]

for cimg in card_images:
    card = dict()
    # {
    #     'col': 'green',
    #     'shape': 'rect',
    #     'fill': 'empty',
    #     'cnt': 1,
    # }

    # converting image into grayscale image
    gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_polys = 0

    lst_intensities = []
    fill_intensities = []

    # list for storing names of shapes
    for contour in contours:
        ## Determine color of contour
        # Create a mask image that contains the contour filled in
        mimg = np.zeros_like(cimg)
        cv2.drawContours(mimg, [contour], -1, color=255, thickness=-1)
        pts = np.where(mimg == 255)
        lst_intensities.extend(cimg[pts[0], pts[1]])

        ## Determine fill shade
        mimg = np.zeros_like(cimg)
        cv2.fillPoly(mimg, [contour], color=255)
        cv2.drawContours(mimg, [contour], -1, color=0, thickness=5) # remove border
        pts = np.where(mimg == 255)
        fill_intensities.extend(cimg[pts[0], pts[1]])

        ## Determine Shape
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.003 * cv2.arcLength(contour, True), True)

        # using drawContours() function
        #cv2.drawContours(cimg, [approx], 0, (0, 0, 255), 2)

        total_polys += len(approx)

    card['cnt'] = len(contours)

    min_col = np.min(lst_intensities, axis=0)
    col_chann = np.argmax(min_col)
    if col_chann == 2:  # Red
        card['col'] = 'red'
    elif col_chann == 1:  # Green
        card['col'] = 'green'
    else:  # Blue
        # can still be green. check contrast
        blue_over_green = min_col[0] - min_col[1]
        if blue_over_green > 20:
            card['col'] = 'purple'
        else:
            card['col'] = 'green'

    avg_fill = np.linalg.norm(np.mean(fill_intensities, axis=0))
    if avg_fill < 250:
        card['fill'] = 'filled'
    elif avg_fill < 340:
        card['fill'] = 'dotted'
    else:
        card['fill'] = 'empty'
    #card['avg_fill'] = avg_fill

    avg_polys = total_polys / len(contours)

    if avg_polys < 10:
        card['shape'] = 'rect'
    elif avg_polys < 22:
        card['shape'] = 'ellipse'
    else:
        card['shape'] = 'squiggely'

    # slice_center = cimg[:, cimg.shape[1]//2, 0]

    # plt.plot(np.diff(np.convolve(slice_center, np.ones(30))))
    # plt.show()

    # cimg = cv2.medianBlur(cimg, 3)
    # card_edges = cv2.Canny(cimg, 0, 250)
    # contours, hierarchy = cv2.findContours(card_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(cimg, contours, -1, (255, 255, 0))

    for i, k in enumerate(card.keys()):
        cv2.putText(cimg, f'{k}={card[k]}', [10, 20 + i * 20], fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=(0, 0, 0))

# Load the first image to get the dimensions
# img_height, img_width = card_images[0].shape
img_height, img_width, _ = card_images[0].shape

# Determine the number of rows and columns in the grid
grid_rows = 6
grid_cols = 2

# Create a blank image to store the grid
grid_height = img_height * grid_cols
grid_width = img_width * grid_rows
grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
# grid_image = np.zeros((grid_height, grid_width), dtype=np.uint8)

# Fill the grid with the images
for i, img in enumerate(card_images[:12]):
    row = i // grid_cols
    col = i % grid_cols
    # print(row*img_height)
    # print(row*img_height+img_height)
    # print(col*img_width)
    # print(col*img_width+img_width)

    # edges_grid = cv2.Canny(img, 0, 250)
    # edges_grid = np.stack([edges_grid, edges_grid, edges_grid], axis=-1)

    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # grid_image[col * img_height:col * img_height + img_height, row * img_width:row * img_width + img_width] = img
    grid_image[col * img_height:col * img_height + img_height, row * img_width:row * img_width + img_width] = img

# Show the grid
cv2.imshow('Grid', grid_image)

cv2.imshow('mask', mask)
cv2.imshow('edges', edges)
cv2.imshow('cont', img_cont)
cv2.imshow('img', input_img)
cv2.waitKey()
cv2.destroyAllWindows()
