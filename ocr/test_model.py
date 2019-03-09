import cv2
import keras
import numpy as np
from keras.models import load_model

model = load_model('chars74k.hdf5')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

img_rows, img_cols = 28, 28

for i in range(81):
    fp = f'thresh_blocks/{i}.png'
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    # img = cv2.fastNlMeansDenoising(img, None, 9, 13)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # img = cv2.bitwise_not(img)
    img = cv2.resize(img, (img_rows, img_cols), cv2.INTER_LANCZOS4)
    img = np.array([img])
    img = img.reshape(img.shape[0], img_rows, img_cols, 1)
    # img = img.astype('float32')
    # img /= 255
    classes = model.predict_classes(img)
    print(f'{i}.png: {classes[0]}')
    # cv2.imshow('x', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
