import cv2
import numpy as np

img_rgb = cv2.imread('assets/img2.jpg', cv2.IMREAD_COLOR)


def process(img):
    kernel = np.ones((2, 2), np.uint8)
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(greyscale, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    denoise = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(denoise, kernel, iterations=1)
    return dilated


def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])  # Getting rid of extra dimenstions

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners


def transform(pts, img):
    pts = np.float32(pts)
    top_l, top_r, bot_l, bot_r = pts[0], pts[1], pts[2], pts[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))

    dim = np.array(([0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]), dtype='float32')
    matrix = cv2.getPerspectiveTransform(pts, dim)
    warped = cv2.warpPerspective(img, matrix, (width, height))
    return warped


processed = process(img_rgb)
corners = get_corners(processed)
warped = transform(corners, img_rgb)
cv2.imshow('x', warped)
cv2.imshow('y', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
