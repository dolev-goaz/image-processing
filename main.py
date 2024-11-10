from binary_error_diffusion import binary_error_diffusion_image
from binary_image import binary_image
from dithering import dither_image
from gray_scale import image_to_grayscale
import numpy as np
import cv2

import os

OUTPUT_FOLDER_NAME = "output"

print("Start of program")
img = cv2.imread("shaked.jpg")

grayscaled_img = image_to_grayscale(img)
dithered_img = dither_image(grayscaled_img)
binary_img = binary_image(grayscaled_img)
binary_diffused_img = binary_error_diffusion_image(grayscaled_img)

mappings = {
    'grayscaled': grayscaled_img,
    'dithered': dithered_img,
    'binary': binary_img,
    'binary_diffused': binary_diffused_img
}
for name, img in mappings.items():
    cv2.imshow(name, img)

os.mkdir(OUTPUT_FOLDER_NAME)
for name, img in mappings.items():
    cv2.imwrite(f"{OUTPUT_FOLDER_NAME}/{name}.png", img)

cv2.waitKey(0)
cv2.destroyAllWindows()