from keras.models import load_model
from sys import argv
import os
from models.model import make_model
from matplotlib.image import imread
from skimage.transform import resize
import numpy as np

MODEL_PATH = "models/butterfly_classifier.h5"
BATCH_SIZE = 128
EPOCHS = 12
TEST_SIZE = 0.2

try:
    model = load_model(MODEL_PATH)
except OSError:
    print(f"Model not found at {MODEL_PATH}! Training a new model...")
    make_model()
    model = load_model(MODEL_PATH)
try:
    path = argv[1]
    image_array = resize(imread(path), (128, 128, 3))
    predictions = model.predict(np.array([image_array]))[0]
    pred_index = np.where(predictions == np.amax(predictions))[0][0]
except IndexError:
    print("Please provide a path to an image!")
except FileNotFoundError:
    print("We did not find an image at that path!")

get_info = lambda num: list(map(lambda s: s.strip(), open(f"data/leedsbutterfly/descriptions/0{num}.txt").readlines()))

butterfly_key = {
    0: get_info("01"),
    1: get_info("02"),
    2: get_info("03"),
    3: get_info("04"),
    4: get_info("05"),
    5: get_info("06"),
    6: get_info("07"),
    7: get_info("08"),
    8: get_info("09"),
    9: get_info("10")
}

print(butterfly_key[pred_index])