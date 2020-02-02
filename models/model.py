import keras as K
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols, 3)

DATA_DIR = "../data/"
IMAGE_ARRAYS = DATA_DIR + "image_arrays.npy"
LABEL_ARRAYS = DATA_DIR + "label_arrays.npy"

X = np.load(IMAGE_ARRAYS, allow_pickle=True)
y = np.load(LABEL_ARRAYS, allow_pickle=True)

encoder = LabelBinarizer()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = np.stack(x_train)
X_test = np.stack(x_test)

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
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=K.losses.categorical_crossentropy,
              optimizer=K.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("butterfly_classifier.h5")
