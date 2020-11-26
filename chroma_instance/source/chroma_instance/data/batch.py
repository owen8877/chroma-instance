from typing import List

import numpy as np


class Instances:
    def __init__(self):
        self.l = []  # dim=(m, m, 1), m=224
        self.ab = []  # dim=(m, m, 2), m=224
        self.bbox = []  # dim=(4, c)
        self.mask = []  # dim=(m, m, c), m=224


class ResizedImages:
    def __init__(self):
        self.l = []  # dim=(m, m, 1), m=224
        self.ab = []  # dim=(m, m, 2), m=224


class Images:
    def __init__(self):
        self.full = []  # dim=(w, h, 3)
        self.l = []  # dim=(w, h)


class Batch:
    def __init__(self):
        self.resized_images = ResizedImages()
        self.images = Images()
        self.instances = Instances()
        self.file_names: List[str] = []

    def normalize(self):
        self.resized_images.l = np.asarray(self.resized_images.l) / 255  # values between 0 and 1
        self.resized_images.ab = np.asarray(self.resized_images.ab) / 255
        self.images.full = np.asarray(self.images.full)
        self.images.l = np.asarray(self.images.l) / 255
        self.instances.l = np.asarray(self.instances.l) / 255  # values between 0 and 1
        self.instances.ab = np.asarray(self.instances.ab) / 255
