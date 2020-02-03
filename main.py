from keras.models import load_model

from models.model import make_model

MODEL_PATH = "models/butterfly_classifier.h3"
BATCH_SIZE = 128
EPOCHS = 12
TEST_SIZE = 0.2

try:
    model = load_model(MODEL_PATH)
except OSError:
    print(f"Model not found at {MODEL_PATH}! Constructing a new model")
    make_model()
    model = load_model(MODEL_PATH)
