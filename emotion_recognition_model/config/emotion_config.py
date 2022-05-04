import os

BASE_PATH = "fer2013"

INPUT_PATH = os.path.join(BASE_PATH, "fer2013.csv")
# Define number of classes: angry, fear, happy, sad, surprise, and neutral
NUM_CLASSES = 6

# Define path to output training, validation and testing
TRAIN_HDF5 = os.path.join(BASE_PATH, "train.hdf5")
VAL_HDF5 = os.path.join(BASE_PATH, "val.hdf5")
TEST_HDF5 = os.path.join(BASE_PATH, "test.hdf5")

# Batch size
BATCH_SIZE = 128

# Path where output logs will be stored
OUTPUT_PATH = "output"