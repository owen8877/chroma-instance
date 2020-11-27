import os

import cv2
import numpy as np
import tensorflow as tf
from keras import applications
from keras.models import load_model
from tqdm import tqdm

from chroma_instance.config.FirstTest import FirstTestConfig
from chroma_instance.data import generator as data
from chroma_instance.model.fusion import FusionModel
from chroma_instance.util import reconstruct, deprocess_float2int


def sample_images(config, test_data):
    avg_cost = 0
    avg_cost2 = 0
    avg_cost3 = 0
    avg_ssim = 0
    avg_psnr = 0

    VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)

    save_path = os.path.join(config.MODEL_DIR, config.PRETRAINED)
    fusion_model = FusionModel(config)
    fusion_model.combined.load_weights(save_path)

    total_batch = int(test_data.size / test_data.batch_size)
    print(f"number of images to colorize: {test_data.size}")
    print(f"total number of batches to colorize: {total_batch}")
    for b in tqdm(range(total_batch)):
        batch = test_data.generate_batch()
        batch_X = batch.resized_images.l
        batch_Y = batch.resized_images.ab

        input_param = [batch_X, batch.instances.l, batch.instances.bbox, batch.instances.mask]

        pred_Y, _, _ = fusion_model.combined.predict(input_param)
        predictVGG = VGG_modelF.predict(np.tile(batch.resized_images.l, [1, 1, 1, 3]))
        loss = fusion_model.combined.evaluate(input_param, [batch.resized_images.ab, predictVGG, np.array((test_data.batch_size,))], verbose=0)
        avg_cost += loss[0]
        avg_cost2 += loss[1]
        avg_cost3 += loss[2]

        int_batch_X = deprocess_float2int(batch_X)
        int_batch_Y = deprocess_float2int(batch_Y)
        int_pred_Y = deprocess_float2int(pred_Y)
        for i in range(test_data.batch_size):
            original_resized_img = reconstruct(int_batch_X[i], int_batch_Y[i])
            predicted_img = reconstruct(int_batch_X[i], int_pred_Y[i])
            ssim = tf.keras.backend.eval(tf.image.ssim(tf.convert_to_tensor(original_resized_img, dtype=tf.float32),
                                                       tf.convert_to_tensor(predicted_img, dtype=tf.float32),
                                                       max_val=255))
            psnr = tf.keras.backend.eval(tf.image.psnr(tf.convert_to_tensor(original_resized_img, dtype=tf.float32),
                                                       tf.convert_to_tensor(predicted_img, dtype=tf.float32),
                                                       max_val=255))
            avg_ssim += ssim
            avg_psnr += psnr

            height, width, _ = batch.images.full[i].shape
            predicted_full_ab = cv2.resize(int_pred_Y[i], (width, height))
            predResult = reconstruct(batch.images.l[i], predicted_full_ab)
            save_path = os.path.join(config.OUT_DIR, f"{psnr:4.8f}_{batch.file_names[i][:-4]}_psnr_reconstructed.jpg")
            cv2.imwrite(save_path, np.concatenate((predResult, batch.images.full[i])))
            print("Batch " + str(b) + "/" + str(total_batch))
            print(psnr)

    print(" ----------  loss =", "{:.8f}------------------".format(avg_cost / total_batch))
    print(" ----------  upsamplingloss =", "{:.8f}------------------".format(avg_cost2 / total_batch))
    print(" ----------  classification_loss =", "{:.8f}------------------".format(avg_cost3 / total_batch))
    print(" ----------  ssim loss =", "{:.8f}------------------".format(avg_ssim / test_data.size))
    print(" ----------  psnr loss =", "{:.8f}------------------".format(avg_psnr / test_data.size))


if __name__ == '__main__':
    config = FirstTestConfig('fusion', ROOT_DIR='../../')
    config.PRETRAINED = 'fusion_combinedEpoch0.h5'
    test_data = data.Data(config.TEST_DIR, config)
    sample_images(config, test_data)
