import cv2
import numpy as np
import numpy.matlib

def binary_error_diffusion_image(gray_scale: cv2.typing.MatLike):
    h, w = gray_scale.shape[:2]
    errors = np.zeros((h, w))
    out = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            value = gray_scale[y, x] + errors[y, x]
            if value < 128:
                out[y, x] = 0
            else:
                out[y, x] = 255
                
            diff = value - out[y,x]
            
            if y + 1 < h:
                errors[y + 1, x] += 3/8 * diff
                
            if x + 1 < w:
                errors[y, x + 1] += 3/8 * diff
                
            if y + 1 < h and x + 1 < w:
                errors[y + 1, x + 1] += 1/8 * diff
    return out.astype(np.uint8)
    
    
            