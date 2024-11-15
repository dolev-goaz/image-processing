from binary_error_diffusion import binary_error_diffusion_image
from binary_image import binary_image
from dithering import dither_image
from gray_scale import image_to_grayscale
from colored_error_diffusion import colored_error_diffusion
import cv2

import cv_utils

OUTPUT_FOLDER_NAME = "output"

print("Start of program")
img = cv2.imread("Lenna.png", cv2.IMREAD_COLOR)
grayscaled_img = image_to_grayscale(img)

mappings = {
    'original': img,
    'color diffused': colored_error_diffusion(img),
    'gray scaled': grayscaled_img,
    # 'dithered': dither_image(grayscaled_img),
    # 'binary': binary_image(grayscaled_img),
    'binary diffused': binary_error_diffusion_image(grayscaled_img)
}

cv_utils.display_images(images=mappings.values(), titles=mappings.keys())


for name, img in mappings.items():
    if name == 'original':
        continue
    file_name = f"{OUTPUT_FOLDER_NAME}/{name}.png"
    cv_utils.save_image(img, file_name)


cv2.waitKey(0)
cv2.destroyAllWindows()