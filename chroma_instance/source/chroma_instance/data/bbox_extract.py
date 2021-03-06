import os

import numpy as np
import skimage.io
import tensorflow as tf
from tqdm import tqdm

import mrcnn.model as model_lib
from chroma_instance.config import FirstTest
from mrcnn import utils, coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 8


# def extract_bbox(config, dir, strategy):
#     inference_config = InferenceConfig()
#     with strategy.scope():
#         model = model_lib.MaskRCNN(mode="inference", model_dir=config.MODEL_DIR, config=inference_config)
#         model.load_weights(config.COCO_MODEL_PATH, by_name=True)
#
#     os.makedirs(f'{config.DATA_DIR}/{dir}_bbox', exist_ok=True)
#     dir_path = os.path.join(config.DATA_DIR, dir)
#
#     files = list(os.listdir(dir_path))
#     batch_size = inference_config.BATCH_SIZE
#     total_batch = (len(files)-1) // batch_size + 1
#     for j in tqdm(range(total_batch)):
#         subfiles = [files[(j*batch_size + i) % len(files)] for i in range(batch_size)]
#         images = [skimage.io.imread(os.path.join(dir_path, file)) for file in subfiles]
#         results = model.detect(images)
#
#         for i in range(batch_size):
#             r = results[i]
#             object_n = len(r['rois'])
#             object_n = min(4, object_n)
#             if object_n >= 1:
#                 mask = r['masks'][:, :, 0].astype(np.int8)
#                 for i in range(1, object_n):
#                     mask += r['masks'][:, :, i].astype(np.int8) * (i + 1)
#                 rois = r['rois'][:object_n]
#             else:
#                 mask = []
#                 rois = []
#
#             np.savez(f'{config.DATA_DIR}/{dir}_bbox/{subfiles[i]}.npz', object_n=object_n, mask=mask, rois=rois)


def extract_bbox(config, dir, strategy):
    inference_config = InferenceConfig()
    with strategy.scope():
        model = model_lib.MaskRCNN(mode="inference", model_dir=config.MODEL_DIR, config=inference_config)
        model.load_weights(config.COCO_MODEL_PATH, by_name=True)

    os.makedirs(f'{config.DATA_DIR}/{dir}_bbox', exist_ok=True)
    dir_path = os.path.join(config.DATA_DIR, dir)

    for file in tqdm(list(os.listdir(dir_path))):
        images = [skimage.io.imread(os.path.join(dir_path, file))]
        results = model.detect(images)

        r = results[0]
        object_n = len(r['rois'])
        object_n = min(4, object_n)
        if object_n >= 1:
            mask = r['masks'][:, :, 0].astype(np.int8)
            for i in range(1, object_n):
                mask += r['masks'][:, :, i].astype(np.int8) * (i + 1)
            rois = r['rois'][:object_n]
        else:
            mask = []
            rois = []

        np.savez(f'{config.DATA_DIR}/{dir}_bbox/{file}.npz', object_n=object_n, mask=mask, rois=rois)


if __name__ == '__main__':
    config = FirstTest.FirstTestConfig('bbox_extract', ROOT_DIR='../../../')
    config.COCO_MODEL_PATH = os.path.join(config.ROOT_DIR, "weights/mask_rcnn/coco.h5")
    if not os.path.exists(config.COCO_MODEL_PATH):
        utils.download_trained_weights(config.COCO_MODEL_PATH)
    strategy = tf.distribute.get_strategy()

    # extract_bbox(config, config.TRAIN_DIR, strategy)
    extract_bbox(config, config.TEST_DIR, strategy)
