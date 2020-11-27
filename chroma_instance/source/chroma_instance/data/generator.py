import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform

from chroma_instance.config import Val2017, MAX_INSTANCES, FirstTest
from chroma_instance.data.batch import Batch


class Data:
    def __init__(self, dirname, config, limit=None):
        self.dir_path = os.path.join(config.DATA_DIR, dirname)
        self.file_list = list(filter(lambda f: '.jpg' in f, os.listdir(self.dir_path)))
        if limit:
            self.file_list = self.file_list[:limit]
        self.size = len(self.file_list)
        self.batch_size = config.BATCH_SIZE
        assert self.batch_size <= self.size, "The batch size should be smaller or equal to the number of training/test images --> modify it in config.py"
        self.image_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE)
        self.instance_shape = (config.INSTANCE_SIZE, config.INSTANCE_SIZE)

        self.data_index = 0

    def read_img(self, filename):
        img = cv2.imread(filename, 3)
        img_resized = cv2.resize(img, self.image_shape)
        img_lab_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        img_l_resized = img_lab_resized[:, :, :1]
        img_ab_resized = img_lab_resized[:, :, 1:]
        img_l = img_lab[:, :, :1]
        return img_l_resized, img_ab_resized, img_resized, img, img_l

    def read_bbox(self, filename, shape):
        result = np.load(filename)
        y_box = result['rois'][:, [0, 2]]
        x_box = result['rois'][:, [1, 3]]
        y_box_ratio = y_box / shape[0]
        x_box_ratio = x_box / shape[1]

        object_n = result['object_n']
        masks = np.zeros((*self.instance_shape, MAX_INSTANCES), dtype=np.float16)
        for i in range(min(object_n, MAX_INSTANCES)):
            masks[:, :, i] = skimage.transform.resize((result['mask'] == i + 1).astype(float),
                                                      self.instance_shape).astype(np.float16)

        bbox = np.zeros((4, MAX_INSTANCES))
        bbox_full = np.concatenate([y_box_ratio, x_box_ratio], axis=1).T
        if bbox.shape[1] > bbox_full.shape[1]:
            bbox[:, :bbox_full.shape[1]] = bbox_full
        else:
            bbox[:, :] = bbox_full[:, :bbox.shape[1]]
        return bbox, masks, object_n

    def segment(self, img, bbox, object_n):
        l = np.zeros((*self.instance_shape, 1, MAX_INSTANCES), dtype=np.uint8)
        ab = np.zeros((*self.instance_shape, 2, MAX_INSTANCES), dtype=np.uint8)
        for i in range(min(object_n, MAX_INSTANCES)):
            y_box = np.floor(bbox[:2, i] * self.image_shape[0]).astype(int)
            x_box = np.floor(bbox[2:, i] * self.image_shape[1]).astype(int)

            instance = cv2.cvtColor(cv2.resize(img[y_box[0]:y_box[1], x_box[0]:x_box[1], :], self.instance_shape),
                                    cv2.COLOR_BGR2Lab)
            l[:, :, :, i] = instance[:, :, :1]
            ab[:, :, :, i] = instance[:, :, 1:]

        return l, ab

    def generate_batch(self) -> Batch:
        batch = Batch()
        for i in range(self.batch_size):
            batch.file_names.append(self.file_list[self.data_index])

            filepath = os.path.join(self.dir_path, self.file_list[self.data_index])
            img_l_resized, img_ab_resized, img_resized, img_full, img_l = self.read_img(filepath)
            batch.resized_images.l.append(img_l_resized)
            batch.resized_images.ab.append(img_ab_resized)
            batch.images.full.append(img_full)
            batch.images.l.append(img_l)

            bbox_filepath = f'{self.dir_path}_bbox/{self.file_list[self.data_index]}.npz'
            bbox, mask, object_n = self.read_bbox(bbox_filepath, img_full.shape)
            batch.instances.bbox.append(bbox)
            batch.instances.mask.append(mask)
            batch.object_n.append(object_n)

            instance_l, instance_ab = self.segment(img_resized, bbox, object_n)
            batch.instances.l.append(instance_l)
            batch.instances.ab.append(instance_ab)

            self.data_index = (self.data_index + 1) % self.size

        batch.normalize()
        return batch


if __name__ == '__main__':
    config = FirstTest.FirstTestConfig(ROOT_DIR='../../../')
    data = Data(config.TRAIN_DIR, config)

    batch = data.generate_batch()

    plt.figure(1)
    for i in range(config.BATCH_SIZE):
        plt.subplot(2, 3, i + 1)
        plt.imshow(cv2.cvtColor(batch.images.full[i], cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.show()

    j = 0
    plt.figure(2)
    for i in range(config.BATCH_SIZE):
        plt.subplot(2, 3, i + 1)
        # if j < len(batch.object_n):
        plt.imshow(batch.instances.mask[i][:, :, j].astype(np.float32))
    plt.title('Masks')
    plt.show()

    plt.figure(3)
    for i in range(config.BATCH_SIZE):
        plt.subplot(2, 3, i + 1)
        # if j < len(batch.object_n):
        img = np.concatenate([batch.instances.l[i][:, :, :, j], batch.instances.ab[i][:, :, :, j]], axis=2)
        plt.imshow(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_Lab2RGB))
    plt.title(f'Instance L (#{j})')
    plt.show()
