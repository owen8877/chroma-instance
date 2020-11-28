import os
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks_v1 import TensorBoard


def deprocess_float2int(imgs):
    # imgs = imgs * 255
    # imgs[imgs > 255] = 255
    # imgs[imgs < 0] = 0
    # return imgs.astype(np.uint8)
    return np.clip(imgs*255, 0, 255).astype(np.uint8)


def reconstruct_and_save(batchX, predictedY, filename, config):
    result = reconstruct(batchX, predictedY)
    save_reconstructed_img(config, filename, result)


def reconstruct(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result


def save_reconstructed_img(config, filename, result):
    save_results_path = os.path.join(config.OUT_DIR, config.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(save_results_path, f"{filename}_reconstructed.jpg")
    cv2.imwrite(save_path, result)
    return result


def write_log(callback: TensorBoard, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def prepare_logger(batch_size, config):
    # Create log folder if needed.
    log_path = os.path.join(config.LOG_DIR, config.TEST_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    date_ymd = datetime.now().strftime("%Y%m%d")
    return open(os.path.join(log_path, f'{date_ymd}_{batch_size}_{config.NUM_EPOCHS}.txt'), "w")