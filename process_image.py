import cv2
import keras
import numpy as np
from keras.models import load_model


def process(img):
    kernel = np.ones((1, 1), np.uint8)
    greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(greyscale, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=2)
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
    square = max(width, height) // 9 * 9  # Making the image dimensions divisible by 9

    dim = np.array(([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype='float32')
    matrix = cv2.getPerspectiveTransform(pts, dim)
    warped = cv2.warpPerspective(img, matrix, (square, square))
    return warped


def subdivide(img, divisions=9):
    height, _ = img.shape
    cluster = height // divisions
    subdivided = img.reshape(height // cluster, cluster, -1, cluster).swapaxes(1, 2).reshape(-1, cluster, cluster)
    return [i for i in subdivided]


def ocr(img_array, img_rows, img_cols):
    for i in img_array:
        img = cv2.resize(i, (img_rows, img_cols), cv2.INTER_LANCZOS4)
        img = np.array([img])
        img = img.reshape(img.shape[0], img_rows, img_cols, 1)
        img = img.astype('float32')
        img /= 255
        classes = model.predict_classes(img)
        print(classes[0])
        cv2.imshow('img', i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


model = load_model('ocr/chars74k_V02.hdf5')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
img_rows, img_cols = 28, 28

img = cv2.imread('assets/img5.jpg', cv2.IMREAD_GRAYSCALE)
processed = process(img)
corners = get_corners(processed)
warped = transform(corners, processed)
subdivided = subdivide(warped)
ocr(subdivided, img_rows, img_cols)
