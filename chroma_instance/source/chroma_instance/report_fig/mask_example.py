from chroma_instance.config.FirstTest import FirstTestConfig
from chroma_instance.data.generator_single_file import DataSingleFile
import matplotlib.pyplot as plt
import numpy as np

config = FirstTestConfig('final6', ROOT_DIR='../../../')
config.TEST_DIR = "coco_test_2017"
test_data = DataSingleFile(config.TEST_DIR, "000000123057.jpg", config)
batch = test_data.generate_batch()

plt.imshow(batch.instances.mask[0, :, :, 0].astype(np.uint8))
plt.show()