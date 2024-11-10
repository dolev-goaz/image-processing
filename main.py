from binary_image import binary_image
from dithering import dither_image
from gray_scale import image_to_grayscale
import numpy as np
import cv2
print("Start of program")
img = cv2.imread("shaked.jpg")

grayscaled_img = image_to_grayscale(img)
dithered_img = dither_image(grayscaled_img)
binary_img = binary_image(grayscaled_img)

cv2.imshow('grayscaled', grayscaled_img)
cv2.imshow('dithered', dithered_img)
cv2.imshow('binary', binary_img)

cv2.waitKey(0)
cv2.destroyAllWindows()