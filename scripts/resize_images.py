import os
import cv2

dir_input = "images"
dir_output = "images_500x500"

os.makedirs(dir_output, exist_ok=True)

for img_fn in os.listdir(dir_input):
    img = cv2.imread(f"{dir_input}/{img_fn}")
    img = cv2.resize(img, [500, 500])
    cv2.imwrite(f"{dir_output}/{img_fn}", img)
