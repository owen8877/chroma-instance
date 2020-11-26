import os

import machine


class FirstTestConfig:
    def __init__(self, ROOT_DIR='../../'):
        # DIRECTORY INFORMATION
        self.TEST_NAME = "FirstTest"
        IDENTIFIER = "chroma_instance"
        ROOT_DIR = os.path.abspath(ROOT_DIR)
        self.ROOT_DIR = ROOT_DIR
        self.DATA_DIR = os.path.join(ROOT_DIR, 'dataset/')
        self.OUT_DIR = os.path.join(ROOT_DIR, f'result/{IDENTIFIER}')
        self.MODEL_DIR = os.path.join(ROOT_DIR, f'weights/{IDENTIFIER}')
        self.LOG_DIR = os.path.join(ROOT_DIR, f'logs/{IDENTIFIER}')

        for dir in (self.DATA_DIR, self.OUT_DIR, self.MODEL_DIR, self.LOG_DIR):
            os.makedirs(dir, exist_ok=True)

        self.TRAIN_DIR = "train"
        self.TEST_DIR = "test"

        # DATA INFORMATION
        self.IMAGE_SIZE = 224
        self.INSTANCE_SIZE = 224

        # TRAINING INFORMATION
        self.PRETRAINED = "my_model_colorizationEpoch4.h5"
        if machine.is_colab:
            self.NUM_EPOCHS = 5
            self.BATCH_SIZE = 10
        else:
            self.NUM_EPOCHS = 2
            self.BATCH_SIZE = 6
