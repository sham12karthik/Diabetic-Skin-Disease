import cv2
import numpy as np

def resize_image(img, size=(224,224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    # keep pixel range 0-1 float32
    img = img.astype('float32') / 255.0
    return img

def remove_hair(img, kernel_size=17):
    """Simple hair removal via morphological closing + inpainting.
    img: RGB uint8 image
    returns: RGB uint8 image with reduced hair lines
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Black-hat to reveal hair-like dark lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Dilate threshold to cover hair width
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    # Inpaint on RGB image
    inpaint = cv2.inpaint(img, dilated, 1, cv2.INPAINT_TELEA)
    return inpaint

def gaussian_denoise(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)