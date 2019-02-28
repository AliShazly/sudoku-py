import cv2
import numpy as np

img = cv2.imread('assets/puzzle.jpg', 0)
width, height = img.shape
img = cv2.GaussianBlur(img, (width, height), 1)


def process(img):
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    denoise = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(denoise, kernel, iterations=1)
    return dilated


processed = process(img)
contours, hire = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
biggest_square_draw = cv2.drawContours(processed, [contours[0]], -1, (0, 255, 0), 2)
