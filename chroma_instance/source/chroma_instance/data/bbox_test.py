import os

import skimage.io

import mrcnn.model as model_lib
from chroma_instance import configs
from mrcnn import utils, coco, visualize
import numpy as np

config = configs.FirstTestConfig(ROOT_DIR='../../../')
config.COCO_MODEL_PATH = os.path.join(config.ROOT_DIR, "weights/mask_rcnn/coco.h5")
if not os.path.exists(config.COCO_MODEL_PATH):
    utils.download_trained_weights(config.COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train_all', 'truck', 'boat', 'traffic light',
      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
      'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
      'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
      'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
      'kite', 'baseball bat', 'baseball glove', 'skateboard',
      'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
      'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
      'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
      'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
      'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
      'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
      'teddy bear', 'hair drier', 'toothbrush']
file_name = os.path.join(config.DATA_DIR, 'test/5951960966_d4e1cda5d0_z.jpg')
image = skimage.io.imread(file_name)

if os.path.exists('result.npy'):
    results = np.load('result.npy')
else:
    inference_config = InferenceConfig()
    model = model_lib.MaskRCNN(mode="inference", model_dir=config.MODEL_DIR, config=inference_config)
    model.load_weights(config.COCO_MODEL_PATH, by_name=True)
    results = model.detect([image], verbose=1)
    np.save('result.npy', results)

r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])