import cv2
import keras
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.models import load_model

from sudoku import solve


def show(*args):
    for i, j in enumerate(args):
        cv2.imshow(str(i), j)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read(fp, greyscale=True):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR)
    old_height, old_width = img.shape[:2]
    size = 800
    if img.shape[0] >= size:
        aspect_ratio = size / float(old_height)
        dim = (int(old_width * aspect_ratio), size)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    elif img.shape[1] >= size:
        aspect_ratio = size / float(old_width)
        dim = (size, int(old_height * aspect_ratio))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    return img


def process(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(greyscale, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=1)
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


def extract_lines(img):
    length = 12
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = cols // length
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    vertical = np.copy(img)
    rows = vertical.shape[0]
    vertical_size = rows // length
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    def draw_lines(im, pts):
        im = np.copy(im)
        pts = np.squeeze(pts)
        for r, theta in pts:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(im, (x1, y1), (x2, y2), (255, 255, 255), 2)
        return im

    lines = draw_lines(grid, pts)
    mask = cv2.bitwise_not(lines)
    return mask


def extract_digits(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    digits = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if cv2.contourArea(i) > img_area * .0005:  # and round(w / h, 2) in (x / 100 for x in range(60, 91))
            cropped = img[y:y + h, x:x + w]
            digits.append(cropped)
    return digits


def add_border(img_arr):
    digits = []
    for i in img_arr:
        crop_h, crop_w = i.shape[:2]
        try:
            pad_h = int(crop_h / 1.75)
            pad_w = (crop_h - crop_w) + pad_h
            pad_h //= 2
            pad_w //= 2
            border = cv2.copyMakeBorder(i, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            digits.append(border)
        except cv2.error:
            continue
    return digits


def subdivide(img, divisions=9):
    height, _ = img.shape[:2]
    box = height // divisions
    if len(img.shape) > 2:
        subdivided = img.reshape(height // box, box, -1, box, 3).swapaxes(1, 2).reshape(-1, box, box, 3)
    else:
        subdivided = img.reshape(height // box, box, -1, box).swapaxes(1, 2).reshape(-1, box, box)
    return [i for i in subdivided]


def sort_digits(subd_arr, template_arr, img_dims):
    sorted_digits = []
    for idx, img in enumerate(subd_arr):
        if np.sum(img) < 255 * img.shape[0]:
            sorted_digits.append(np.zeros((img_dims, img_dims), dtype='uint8'))
            continue
        for idx2, template in enumerate(template_arr):
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= .9)
            for i in zip(*loc[::-1]):
                if i is not None:
                    sorted_digits.append(template)
                    break
            else:
                continue
            break
    return sorted_digits


def img_to_array(img_arr, img_dims):
    predictions = []
    for i in img_arr:
        resized = cv2.resize(i, (img_dims, img_dims), interpolation=cv2.INTER_LANCZOS4)
        if np.sum(resized) == 0:
            predictions.append(0)
            continue
        array = np.array([resized])
        reshaped = array.reshape(array.shape[0], img_dims, img_dims, 1)
        flt = reshaped.astype('float32')
        flt /= 255
        prediction = model.predict_classes(flt)
        predictions.append(prediction[0])
    puzzle = np.array(predictions).reshape((9, 9))
    return puzzle


def put_solution(img_arr, soln_arr, unsolved_arr):
    solutions = np.array(soln_arr).reshape(81)
    unsolveds = np.array(unsolved_arr).reshape(81)
    paired = list((zip(solutions, unsolveds, img_arr)))
    img_solved = []
    for solution, unsolved, img in paired:
        if solution == unsolved:
            img_solved.append(img)
            continue
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        fnt = ImageFont.truetype('assets/FreeMono.ttf', img_h)
        font_w, font_h = draw.textsize(str(solution), font=fnt)
        draw.text(((img_w - font_w) / 2, (img_h - font_h) / 2 - img_h // 10), str(solution),
                  fill=((0, 127, 255) if len(img.shape) > 2 else 0), font=fnt)
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_solved.append(cv2_img)
    return img_solved


def stitch_img(img_arr, img_dims):
    result = Image.new('RGB' if len(img_arr[0].shape) > 2 else 'L', img_dims)
    box = [0, 0]
    for img in img_arr:
        pil_img = Image.fromarray(img)
        result.paste(pil_img, tuple(box))
        if box[0] + img.shape[1] >= result.size[1]:
            box[0] = 0
            box[1] += img.shape[0]
        else:
            box[0] += img.shape[1]
    return np.array(result)


def inverse_perspective(img, dst_img, pts):
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img


model = load_model('ocr/chars74k_thresh.hdf5')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
img_dims = 28

fp = 'assets/c6.jpg'
img = read(fp, greyscale=False)
processed = process(img)
corners = get_corners(processed)
warped = transform(corners, processed)
mask = extract_lines(warped)
numbers = cv2.bitwise_and(warped, mask)
digits_unsorted = extract_digits(numbers)
digits_subd = subdivide(numbers)
digits_sorted = sort_digits(digits_subd, digits_unsorted, img_dims)
digits_border = add_border(digits_sorted)
puzzle = img_to_array(digits_border, img_dims)
solved = solve(puzzle.copy().tolist())  # Solve function modifies original puzzle var
warped_img = transform(corners, img)
subd = subdivide(warped_img)
subd_soln = put_solution(subd, solved, puzzle)
warped_soln = stitch_img(subd_soln, (warped_img.shape[0], warped_img.shape[1]))
warped_inverse = inverse_perspective(warped_soln, img, np.array(corners))
show(warped_inverse)
