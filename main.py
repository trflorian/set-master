import cv2

import numpy as np

# from inspyrenet_inference import mask_image

IMG_DIR = 'images'
mask_dir = 'results'

CARD_WIDTH = 200
CARD_HEIGHT = 350

img_fn = 'PXL_20230205_150355105'


# img_fn = 'PXL_20230205_150409042'
# img_fn = 'PXL_20230205_150413474'
# img_fn = 'PXL_20230205_150348842'
# img_fn = 'PXL_20230205_150345491.MP'

def load_input_image(image_name: str, resize_lr=(500, 500)):
    image_fn = f'{IMG_DIR}/{image_name}.jpg'

    input_image = cv2.imread(image_fn)
    input_image_lr = cv2.resize(input_image, resize_lr)

    return input_image, input_image_lr


def create_mask(image):
    # TODO remove
    mask = cv2.imread(f'{mask_dir}/{IMG_DIR}_500x500/{img_fn}.png')
    return mask

    # TODO
    # return np.array(mask_image(input_img_fn))


def pre_process_mask(mask):
    # Grayscale mask
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Shrink mask bounds
    mask = cv2.erode(mask, np.ones((3, 3)))

    return mask


def extract_contours_from_mask(mask):
    # Thresholding is better method, finding contours on edge detected image has artifacts
    _, threshold = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
    # edges = cv2.Canny(mask, 0, 250)

    # Extract contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # img_cont = threshold.copy()
    # cv2.drawContours(img_cont, contours, -1, [255, 255, 0], thickness=5)

    return contours


def cards_from_contours(contours):
    all_cards = []

    for cont in contours:
        # Approximate the contour
        epsilon = 0.05 * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)

        area = cv2.contourArea(approx)
        # TODO
        # cv2.drawContours(img_cont, [approx], -1, [255, 255, 0], thickness=2)

        if len(approx) == 4 and area >= 400:
            all_cards.append(approx)
        else:
            print(f'disposing card... {area=} {len(approx)=}')

    print(f'Found {len(all_cards)} Cards!')

    if len(all_cards) < 12:
        print('Did not find all cards!')
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
        all_cards = [all_cards[i] for i in sorted_indices]

    # cv2.drawContours(input_img, cards, -1, (0, 255, 255), 2)
    # cv2.drawContours(input_img, deleted_cards, -1, (0, 0, 255), 2)

    return all_cards


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # TODO visualize corner points
    # for p in rect:
    #     x = int(p[1] / scale_x)
    #     y = int(p[0] / scale_y)
    #     # image[y,x]=[0,0,255]
    #
    #     cv2.circle(input_img, (y, x), radius=5, color=(0, 0, 255), thickness=-1)
    #     # cv2.circle(image, (x, y), radius=0, color=(0, 0, 255), thickness=-1)

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def hr_card_images_from_lr_contours(card_contours, hr_image, hr_scale_factor, card_size):
    card_images = []
    for card_cnt in card_contours:
        #
        card_cnt = card_cnt.reshape(-1, 2).astype(np.float32)

        # scale cards to OG image size
        card_cnt[:, 1] *= hr_scale_factor[0]
        card_cnt[:, 0] *= hr_scale_factor[1]

        # print(card)
        unwarped = four_point_transform(hr_image, card_cnt)

        # cv2.imshow(f'card {card}', unwarped)

        if unwarped.shape[0] < unwarped.shape[1]:
            # print('rotating')
            unwarped = cv2.rotate(unwarped, cv2.ROTATE_90_CLOCKWISE)

        resized_card = cv2.resize(unwarped, card_size)

        resized_card = cv2.normalize(resized_card, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

        card_images.append(resized_card)

    return card_images


def extract_card_properties(card_images, card_size):
    card_properties = []
    card_area = card_size[0] * card_size[1]
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
        _, threshold = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        # using a findContours() function
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_polys = 0
        convex_polys = 0

        shape_cnt = 0

        lst_intensities = []
        fill_intensities = []

        # list for storing names of shapes
        for contour in contours:
            area = cv2.contourArea(contour)

            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M["m10"] / M["m00"]) / card_size[0]
                cY = int(M["m01"] / M["m00"]) / card_size[1]
            else:
                cX = 0
                cY = 0

            cf = area / card_area
            if cf < 0.08 or cf > 0.2 or cY < 0.2 or cY > 0.8 or cX < 0.4 or cX > 0.6:
                # print(f'skipping shape on card with area: {cf:.4f}')
                continue

            shape_cnt += 1

            ## Determine color of contour
            # Create a mask image that contains the contour filled in
            mimg = np.zeros_like(cimg)
            cv2.drawContours(mimg, [contour], -1, color=255, thickness=-1)
            pts = np.where(mimg == 255)
            lst_intensities.extend(cimg[pts[0], pts[1]])

            ## Determine fill shade
            mimg = np.zeros_like(cimg)
            cv2.fillPoly(mimg, [contour], color=255)
            cv2.drawContours(mimg, [contour], -1, color=0, thickness=16)  # remove border
            pts = np.where(mimg == 255)
            fill_intensities.extend(cimg[pts[0], pts[1]])

            ## Determine Shape
            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.008 * cv2.arcLength(contour, True), True)

            convexHull = cv2.convexHull(approx)

            # using drawContours() function
            # cv2.drawContours(cimg, [convexHull], 0, (0, 0, 255), 2)
            cv2.drawContours(cimg, [approx], 0, (0, 0, 255), 2)

            convex_polys += len(convexHull)
            total_polys += len(approx)

        card['cnt'] = shape_cnt

        if len(lst_intensities) > 0:
            min_col = np.min(lst_intensities, axis=0)
        else:
            min_col = [0, 0, 0]

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

        if len(fill_intensities) == 0:
            avg_fill = 0
            min_fill = 0
        else:
            avg_fill = np.mean(np.linalg.norm(fill_intensities, axis=1))
            min_fill = np.min(np.linalg.norm(fill_intensities, axis=1))
        diff_fill = avg_fill - min_fill
        if diff_fill > 100:
            card['fill'] = 'dotted'
        elif avg_fill < 250:
            card['fill'] = 'filled'
        else:
            card['fill'] = 'empty'

        # card['avg_fill'] = avg_fill
        # card['diff_fill'] = diff_fill

        if shape_cnt > 0:
            avg_polys = total_polys / shape_cnt
            avg_conv = convex_polys / shape_cnt
        else:
            avg_polys = 0
            avg_conv = 0
        # card['avg_polys'] = int(avg_polys)
        # card['avg_conv'] = int(avg_conv)

        if avg_polys < 10:
            card['shape'] = 'rect'
        elif avg_polys - avg_conv < 2:
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

        card_properties.append(card)

    return card_properties


def create_card_grid(card_images, card_size, grid_rows, grid_cols):
    w = card_size[0]
    h = card_size[1]

    # Create a blank image to store the grid
    grid_height = h * grid_cols
    grid_width = w * grid_rows
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
        grid_image[col * h:col * h + h, row * w:row * w + w] = img

    return grid_image


def main():
    card_size = (CARD_WIDTH, CARD_HEIGHT)

    hr_image, input_img = load_input_image(img_fn)

    scale_x = hr_image.shape[0] / input_img.shape[0]
    scale_y = hr_image.shape[1] / input_img.shape[1]

    # Mask is LR
    mask = create_mask(hr_image)
    mask = pre_process_mask(mask)

    # Contours are LR
    all_contours = extract_contours_from_mask(mask)
    # previously cards
    card_contours = cards_from_contours(all_contours)

    card_images = hr_card_images_from_lr_contours(
        card_contours=card_contours,
        hr_image=hr_image,
        hr_scale_factor=(scale_x, scale_y),
        card_size=card_size,
    )

    card_properties = extract_card_properties(
        card_images=card_images,
        card_size=card_size,
    )

    card_grid = create_card_grid(
        card_images=card_images,
        card_size=card_size,
        grid_rows=6,
        grid_cols=2
    )

    # Show the grid
    cv2.imshow('Grid', card_grid)

    cv2.imshow('mask', mask)
    # cv2.imshow('threshold', threshold)

    # cv2.imshow('edges', edges)

    # cv2.imshow('cont', img_cont)
    cv2.imshow('img', input_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
