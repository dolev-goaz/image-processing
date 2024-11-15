from binary_error_diffusion import binary_error_diffusion_image
from binary_image import binary_image
from dithering import dither_image
from gray_scale import image_to_grayscale
from colored_error_diffusion import colored_error_diffusion
import cv2

import cv_utils

import os

OUTPUT_FOLDER_NAME = "output"

print("Start of program")
img = cv2.imread("shaked.jpg", cv2.IMREAD_COLOR)
grayscaled_img = image_to_grayscale(img)

mappings = {
    # 'grayscaled': grayscaled_img,
    'color_diffused': colored_error_diffusion(img),
    # 'dithered': dither_image(grayscaled_img),
    # 'binary': binary_image(grayscaled_img),
    # 'binary_diffused': binary_error_diffusion_image(grayscaled_img)
}

cv_utils.display_images(images=[img, grayscaled_img, mappings['color_diffused']], titles=['Source', 'GrayScaled', 'Diffused'])


for name, img in mappings.items():
    cv2.imshow(name, img)


# if not os.path.exists(OUTPUT_FOLDER_NAME):
#     os.mkdir(OUTPUT_FOLDER_NAME)
# for name, img in mappings.items():
#     cv2.imwrite(f"{OUTPUT_FOLDER_NAME}/{name}.png", img)


cv2.waitKey(0)
cv2.destroyAllWindows()