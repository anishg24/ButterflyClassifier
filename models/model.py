import keras as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 12
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols, 3)

df = pd.read_csv("../data/dataframe.csv")

# temp_X = np.array(df["Image Arrays"])
# y = np.array(df["Name"])
#
# temp_X_train, temp_X_test, y_train, y_test = train_test_split(temp_X, y, test_size=0.3)


# def breakdown(arr):
#     X = []
#     for category in arr:
#         for photo in category:
#             X.append(photo)
#     return np.array(X)
#
#
# X_train, X_test = breakdown(temp_X_train), breakdown(temp_X_test)

X = np.array(df["Image Arrays"])
y = np.array(df["Name"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

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
          validation_data=(X_test, y_test)
          )
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
