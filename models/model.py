# import keras as K
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv("../data/dataframe.csv")

y = df["Name"]
X = np.array(df["Image Arrays"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# X_train = X_train.reshape(60000,28,28,1)
# X_test = X_test.reshape(10000,28,28,1)

print(X_train[0])