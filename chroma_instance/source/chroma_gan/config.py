import os
import machine

# DIRECTORY INFORMATION
TEST_NAME = "FirstTest"
ROOT_DIR = os.path.abspath('../../')
DATA_DIR = os.path.join(ROOT_DIR, 'dataset/')
OUT_DIR = os.path.join(ROOT_DIR, 'result/chroma_gan')
MODEL_DIR = os.path.join(ROOT_DIR, 'weights/chroma_gan')
LOG_DIR = os.path.join(ROOT_DIR, 'logs/chroma_gan/')

for dir in (DATA_DIR, OUT_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(dir, exist_ok=True)

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
IMAGE_SIZE = 224

# TRAINING INFORMATION
PRETRAINED = "imagenet.h5"
if machine.is_colab:
    NUM_EPOCHS = 5
    BATCH_SIZE = 10
else:
    NUM_EPOCHS = 2
    BATCH_SIZE = 6

