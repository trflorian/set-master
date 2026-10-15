import cv2
import os
import matplotlib.pyplot as plt

import numpy as np

img_dir = "images"

img_files = os.listdir(img_dir)
img_paths = [f"{img_dir}/{f}" for f in img_files]

for img_fn in img_paths[3:]:
    img = cv2.imread(img_fn)
    img = cv2.resize(img, [400, 400])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray, 11)

    edges = cv2.Canny(blur, 100, 250)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # ret3, th3 = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont_img = img.copy()
    cv2.drawContours(cont_img, contours, -1, [255, 255, 0], thickness=5)

    approx_cont_img = img.copy()
    wrong_cont_img = img.copy()

    areas = []
    cards = []

    for cont in contours:
        # Approximate the contour
        epsilon = 0.1 * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)

        area = cv2.contourArea(approx)

        # Check if the contour has 4 corners
        if len(approx) == 4 and area >= 500:
            areas.append(area)
            cards.append(approx)
        else:
            cv2.drawContours(wrong_cont_img, [approx], 0, (0, 0, 255), 2)

    print(f"Found {len(cards)} Cards!")
    cv2.drawContours(approx_cont_img, cards, -1, (0, 0, 255), 2)

    if len(cards) != 12:
        cv2.imshow("Photo", img)
        cv2.imshow("Gray", gray)
        cv2.imshow("Edges", edges)
        cv2.imshow("Contours", cont_img)
        cv2.imshow("Wrong Contours", wrong_cont_img)
        cv2.imshow("Approx Contours", approx_cont_img)
        cv2.imshow("Thresh", th3)

        cv2.waitKey()
        cv2.destroyAllWindows()
