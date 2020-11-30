import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from tqdm import tqdm
from lpips_tensorflow.lpips_tf import lpips

class Metric:
    def __init__(self, name, fct):
        self.name = name
        self.fct = fct
        self.content = np.array([])

    def add(self, img_true, img_pred, session):
        np.append(self.content, session.run(self.fct(img_true, img_pred)))

    def average(self):
        return self.content.mean()


class Network:
    def __init__(self, name, filename_decorator):
        self.name = name
        self.filename_decorator = filename_decorator
        self.datasets = {}
        for dataset in 'places205', 'imagenet_ctest', 'coco_test_2017':
            self.datasets[dataset] = [
                Metric('psnr', lambda img_true, img_pred: tf.image.psnr(original_img, predicted_img, max_val=255)),
                Metric('ssim', lambda img_true, img_pred: tf.image.ssim(original_img, predicted_img, max_val=255)),
                Metric('lpips', lambda img_true, img_pred: lpips(tf.cast(img_true, tf.float32) / 255.0, tf.cast(img_pred, tf.float32) / 255.0, model='net-lin', net='alex')),
            ]


networks = [
    Network('chroma_gan', lambda f: f + 'psnr_reconstructed.jpg'),
    Network('fusion_2obj', lambda f: f + '_reconstructed.jpg'),
    Network('fusion_2obj_huber', lambda f: f + '_reconstructed.jpg'),
    Network('instcolorization', lambda f: f + 'psnr_reconstructed.jpg'),
]

with tf.Session() as session:
    for network in networks:
        print(f'Evaluating {network.name}')
        for dataset_name, dataset in network.datasets.items():
            print(f'On dataset {dataset_name}')
            for filename in tqdm(os.listdir(f'../dataset/{dataset_name}')):
                original_img = tf.convert_to_tensor(imread(f'../dataset/{dataset_name}/{filename}'))
                for metric in dataset:
                    predicted_img = tf.convert_to_tensor(imread(f'../result/{network.name}/{dataset_name}/{network.filename_decorator(filename)}'))
                    metric.add(original_img, predicted_img, session)

for network in networks:
    for dataset_name, dataset in network.datasets.items():
        for metric in dataset:
            print(f'{network.name} on {dataset_name}: {metric.name}={metric.average()}')
