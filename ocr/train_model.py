import json
from random import shuffle

import cv2
import keras
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

batch_size = 64
num_classes = 9
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
x_train = []
for i in range(8235):
    fp = f'chars_train/{i}.png'
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 231, 0)
    img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
    x_train.append(img)

x_test = []
for i in range(909):
    fp = f'chars_test/{i}.png'
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 231, 0)
    img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
    x_test.append(img)

with open('labels_train.json') as f:
    y_train = json.load(f)
    y_train = [i - 1 for i in y_train]

with open('labels_test.json') as f:
    y_test = json.load(f)
    y_test = [i - 1 for i in y_test]

y_train, y_test, x_train, x_test = np.array(y_train), np.array(y_test), np.array(x_train), np.array(x_test)

ind_list_train = [i for i in range(x_train.shape[0])]
shuffle(ind_list_train)
x_train = x_train[ind_list_train, :, :, ]
y_train = y_train[ind_list_train,]

ind_list_test = [i for i in range(x_test.shape[0])]
shuffle(ind_list_test)
x_test = x_test[ind_list_test, :, :, ]
y_test = y_test[ind_list_test,]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model.hdf5')
