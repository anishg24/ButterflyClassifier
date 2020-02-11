# Butterfly Classifier
This project is made by Anish Govind. Other projects can be found at my [GitHub](https://github.com/anishg24).
Huge thank you to the contributors behind this [dataset](https://www.kaggle.com/veeralakrishna/butterfly-dataset) as without it
this project wouldn't be published

![GitHub followers](https://img.shields.io/github/followers/anishg24?label=Follow&style=social)

![Status](https://img.shields.io/badge/status-completed-brightgreen?style=flat-square)
![Version](https://img.shields.io/github/v/release/anishg24/ButterflyClassifier?color=orange&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/anishg24/ButterflyClassifier?style=flat-square)

## Project Objective
The purpose of this project is to classify between 10 species of butterflies based of a picture of a butterfly. Butterflies are important and crucial to our ecosystem. 
They are vital in pollination, as they spread pollen when they collected nectar, and the ability to classify between them would lead to knowing which butterflies are more useful in pollination. 

These are the supported species by this dataset and model:

Scientific Name | Common Name
------------ | -------------
Heliconius Erato | Crimson-patched Longwing
Heliconius charitonius | Zebra Longwing
Danaus plexippus | Monarch
Lycaena phlaeas | American Copper
Junonia coenia | Common Buckeye
Vanessa cardui | Painted Lady
Nymphalis antiopa | Mourning Cloak
Papilio cresphontes | Giant Swallowtail
Vanessa atalanta | Red Admiral
Pieris rapae | Cabbage White

Any other species are **not** supported by this [dataset](https://www.kaggle.com/veeralakrishna/butterfly-dataset) and therefore this model!
Running the model based off another species will **not** produce an error, but it will provide an incorrect prediction!

### Methods Used
* Inferential Statistics
* Deep Learning
* Convolutional Neural Networks

### Technologies
* Python
* Pandas
* Jupyter & Matplotlib
* Numpy
* Sci-Kit Learn (for data handling)
* Keras & Tensorflow

## Project Description
This project was developed off the data (provided [here](https://www.kaggle.com/veeralakrishna/butterfly-dataset)) which consisted
of 10 text files and ~800 images of butterflies. The data is marked in the way that the file name is the first 3 digits of the 
image file (a butterfly from `001.txt` would have image `0010001.png`). Armed with this, a couple of iterative sequences is able
to populate a dictionary with 2 pieces of information: Scientific Name and the Image Array.

Initially, the dictionary was structured as such:
```python
my_dict = {
    "Names": ["Monarch", "Crimson-patched Longwing", ...],
    "Scientific Names": ["Danaus plexippus", "Heliconius erato", ...],
    "Image Files": [["0010001.png", ...], ...], # List of lists of image files that relate to the iterated species
    "Image Arrays": [
        [
            [[[...]]], ... # 3-D Image matrices from matplotlib.image.imread. Resized to be 128x128.
        ],
        [
            [[[...]]], ... # 3-D Image matrices from matplotlib.image.imread. Resized to be 128x128.
        ],
        ...
    ]
}
```
And would create a DataFrame similar to the following:

Names | Scientific Names | Image Files | Image Arrays
------------ | ------------- | ------------ | ------------- |
"Monarch" | "Danaus plexippus" | ["0010001.png", ...] | [[[...]]]
"Crimson Patched Longwing" | "Heloconius erato" | ... | [[[...]]]
... | ... | ... | ...

This would be good, for normal database handling, but in terms of running models off this data, it was a terrible way to do this.
Instead, we should try to have each **input** have an **output**, instead of having one **output** containing multiple **inputs**.
Realizing this was when I was developing the model itself, and running into errors like dimension mismatch. Nonetheless I figured it out
and now have a dictionary structured like:

```python
my_dict = {
    "Scientific Name": ["Danaus plexippus", "Danaus plexippus", ..., "Heliconius erato", ...], # Notice how there are multiple occurrences of the same species this time
    "Image Arrays": [[[[...]]], [[[...]]], ..., [[[...]]], ...] # Notice how this is a list of 3-D Image matrices from matplotlib.image.imread resized to be 128x128, 
                                                                # instead of a list of 4-D matrices.
}
```
I wouldn't need to record the image files anywhere, as it isn't used for input, so I dropped it. This would then create a DataFrame similar to:

Scientific Name | Image Arrays
------------ | -------------
"Danaus plexippus" | [[[...]]]
"Danaus plexippus" | [[[...]]]
... | ...
"Heliconius erato" | [[[...]]]
... | ...

Perfect! Now I can map each input of the image array to a specific species! After handling my data the *correct* way, I 
saved it as `.npy` files in the `data/` directory. This would prevent us from having to create a DataFrame every time we want to predict from an image
as I can just call the file later on and store that to my `model.py`. 

I trained the model using Keras and used multiple convolutional neurons to make a reliable model. After training for a few short
minutes, I would save the model into `butterfly_classifier.h5`. Saving the model there would allow us to make predictions quicker
for the same reasons above.

In `main.py` we handle the prediction and other possible errors. This is the part where the user interacts with the program, so you'll
bear witness to multiple checks and balances in the forms of `try/except` blocks. But if we were to allow users to send in whatever image
they want, then we would likely have shape mismatch errors (as in feeding my model an array of size `(1440, 1080, 3)` when the input only accepts `(128, 128, 3)`.
To ensure the user won't break everything, I have added the same way to resize the image down.Also in `main.py` I have added a way to reverse the one-hot
encoding that took place in `model.py` as if the program just returned an index number for the prediction matrix, it would not be a good program.
Through careful analysis (and by that I meant playing in my `Data Processing.ipynb` notebook) I figured out how the array was one-hot
encoded and reversed it to provide the correct species matched to the input image. Adding a way to get user input (by the means of sys.argv)
I have completed the rough outline of the project!

What this project was designed for was to mainly test my knowledge about machine learning and whether or not I can make a 
reliable model based off a dataset. Instead of tackling the MNIST sets, I wanted to do something different but just as similar. Typical
MNIST dataset tutorials give you a preprocessed dataset all in one line of code. Where's the fun in that? Where's the data pre-processing? I opted
for doing all that by myself and it seemed to work in my favor.


## Needs of this project

- More pictures of different species (more important)
- More pictures of the same species (lower priority)

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Download the dataset [here](https://www.kaggle.com/veeralakrishna/butterfly-dataset) and unzip the file in the `data/` folder of this repo. 
   **DO NOT CHANGE THE FOLDER NAME FROM **`leedsbutterfly/`
3. `conda create venv`
4. `conda activate venv`
5. `conda install -r requirements.txt`
6. `python main.py ~/PATH/TO/YOUR/IMAGE`

Note: The `label_arrays.npy`, `image_arrays.npy`, and `butterfly_classifier.h5` files are **not** provided while cloning the repository.
This means that upon the first run, the script will automatically generate those files, but it will consume time and resources.
To get access to these files and quickly run the model, check out the [releases](https://github.com/anishg24/ButterflyClassifier/releases). 
If you do decide to download the files and run the classifier, you don't need
to use the dataset, and can safely delete it.

## Arguments
Argument | Output
------------ | -------------
`~/PATH/TO/YOUR/IMAGE` | ["Scientific Name", "Common Name", "Brief Description"]
`-h` `--help` | Show a help message and exit
`-r` `--retrain` | Retrains the model and runs the new model to predict off the given image. The new model gets saved in a file.
`-b` `--batch_size` | Changes the default batch size from 128 to user input
`-e` `--epochs` | Changes the default epoch size from 12 to user input
`-t` `--test_size` | Changes the default test proportion from 0.2 (20%) to user input.

## To-Do
- [x] Preprocess dataset
- [x] Create the model and save it
- [x] Make a working script to run everything and predict
- [x] Add the ability to retrain the model with new user input
- [ ] Data augmentation to improve accuracy of the models
- [ ] Make a website that allows you to add images or take images of butterflies and tells you what species they are.

## Releases
- 1.0.0 (2/9/2020): First working release. More to come soon :smile:
- 1.2.0 (2/11/2020): Added command line arguments.

## Contributing Members

Creator: [Anish Govind](https://github.com/anishg24@gmail.com)

Ways to contact:
* [E-Mail](anishg24@gmail.com)

**IF YOU FIND ANY ISSUES OR BUGS PLEASE OPEN AN ISSUE**
