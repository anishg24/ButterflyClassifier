import os

import numpy as np
import pandas as pd
from matplotlib.image import imread
from skimage.transform import resize

DATA_DIR = "data/leedsbutterfly"
DESC_DIR = DATA_DIR + "/descriptions/"
IMG_DIR = DATA_DIR + "/images/"
array_save = "image_arrays.npy"
label_save = "label_arrays.npy"


def process():
    my_dict = {
        "Scientific Name": [],
        "Image Files": [],
        "Image Arrays": [],
    }
    for f in os.listdir(DESC_DIR):
        file_dir = DESC_DIR + f"/{f}"
        with open(file_dir, 'r') as d:
            lines = d.readlines()
            for i in os.listdir(IMG_DIR):
                if i.startswith(f[:3]):
                    my_dict["Image Files"].append(i)
                    my_dict["Scientific Name"].append(lines[0].strip())
                    my_dict["Image Arrays"].append(resize(imread(IMG_DIR + i), (128, 128, 3)))

    df = pd.DataFrame.from_dict(my_dict)

    np.save(array_save, df["Image Arrays"].values)
    np.save(label_save, df["Scientific Name"].values)


print(f"Saved the image arrays to {array_save} and the labels to {label_save}")
