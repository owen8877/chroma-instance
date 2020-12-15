import matplotlib.pyplot as plt
import numpy as np
import skimage.transform

from chroma_instance.config.FirstTest import FirstTestConfig
from chroma_instance.data.generator_single_file import DataSingleFile

config = FirstTestConfig('instance', ROOT_DIR='../../../')
config.TEST_DIR = "imagenet_ctest"
test_data = DataSingleFile(config.TEST_DIR, "ILSVRC2012_val_00033944.JPEG", config)
batch = test_data.generate_batch()

for j in range(2):
    plt.imsave(f'../../../../figs/instance{j}_mask.jpg', batch.instances.mask[0, :, :, j].astype(np.uint8))

    shape = batch.images.full[0].shape
    bbox = batch.instances.bbox[0]
    y_0 = int(shape[0] * bbox[0, j])
    y_1 = int(shape[0] * bbox[1, j])
    x_0 = int(shape[1] * bbox[2, j])
    x_1 = int(shape[1] * bbox[3, j])

    cropped = batch.images.full[0][y_0:y_1, x_0:x_1, :]
    resized = skimage.transform.resize(cropped, (224, 224))
    plt.imsave(f'../../../../figs/instance{j}.jpg', resized[:, :, [2, 1, 0]])
