import os

# DIRECTORY INFORMATION
# DATASET = "imagenet" # UPDATE
TEST_NAME = "FirstTest"
ROOT_DIR = os.path.abspath('../../')
DATA_DIR = os.path.join(ROOT_DIR, 'dataset/')
OUT_DIR = os.path.join(ROOT_DIR, 'result/chroma_gan')
MODEL_DIR = os.path.join(ROOT_DIR, 'weights/chroma_gan')
LOG_DIR = os.path.join(ROOT_DIR, 'logs/chroma_gan/')

for dir in (DATA_DIR, OUT_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(dir, exist_ok=True)

TRAIN_DIR = "train"  # UPDATE
TEST_DIR = "test"  # UPDATE

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 1

# TRAINING INFORMATION
PRETRAINED = "imagenet.h5"  # UPDATE
NUM_EPOCHS = 5
