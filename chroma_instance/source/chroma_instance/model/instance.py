import os
from datetime import datetime
from functools import partial

import cv2
import numpy as np
from keras import Model
from keras.callbacks import TensorBoard
from keras.layers import Input
from keras.optimizers import Adam

from chroma_instance.config.FirstTest import FirstTestConfig
from chroma_instance.data.generator import Data
from chroma_instance.model.basic import discriminator_network, RandomWeightedAverage, \
    wasserstein_loss_dummy, gradient_penalty_loss, simple_instance_network
from chroma_instance.util import write_log, deprocess, reconstruct, train

GRADIENT_PENALTY_WEIGHT = 10


class InstanceModel:
    def __init__(self, config):
        instance_shape = (config.INSTANCE_SIZE, config.INSTANCE_SIZE)

        # Creating generator and discriminator
        optimizer = Adam(0.00002, 0.5)

        self.foreground_discriminator = discriminator_network(instance_shape)
        self.foreground_discriminator.compile(loss=wasserstein_loss_dummy, optimizer=optimizer)
        # TODO: the dimension to distinguish different instances are moved towards the batch dimension
        self.foreground_generator = simple_instance_network(instance_shape)
        self.foreground_generator.compile(loss=['mse'], optimizer=optimizer)

        img_l = Input(shape=(*instance_shape, 1))
        img_real_ab = Input(shape=(*instance_shape, 2))

        self.foreground_generator.trainable = False
        img_pred_ab = self.foreground_generator(img_l)
        dis_pred_ab = self.foreground_discriminator([img_pred_ab, img_l])
        dis_real_ab = self.foreground_discriminator([img_real_ab, img_l])

        # Sample the gradient penalty
        img_ab_interp_samples = RandomWeightedAverage()([img_real_ab, img_pred_ab])
        dis_interp_ab = self.foreground_discriminator([img_ab_interp_samples, img_l])
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=img_ab_interp_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'

        # Compile D and G as well as combined
        self.discriminator_model = Model(inputs=[img_l, img_real_ab],
                                         outputs=[dis_real_ab,
                                                  dis_pred_ab,
                                                  dis_interp_ab])

        self.discriminator_model.compile(optimizer=optimizer,
                                         loss=[wasserstein_loss_dummy,
                                               wasserstein_loss_dummy,
                                               partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])

        self.foreground_generator.trainable = True
        self.foreground_discriminator.trainable = False
        self.combined = Model(inputs=[img_l],
                              outputs=[img_pred_ab, dis_pred_ab])
        self.combined.compile(loss=['mse', wasserstein_loss_dummy],
                              loss_weights=[1.0, -0.1],
                              optimizer=optimizer)

        # Monitor stuff
        self.log_path = os.path.join(config.LOG_DIR, config.TEST_NAME)
        self.callback = TensorBoard(self.log_path)
        self.callback.set_model(self.combined)
        self.train_names = ['loss', 'mse_loss', 'kullback_loss', 'wasserstein_loss']
        self.disc_names = ['disc_loss', 'disc_valid', 'disc_fake', 'disc_gp']

        self.test_loss_array = []
        self.g_loss_array = []

    def scatter(self, stacked_l, stacked_ab):
        scattered_l = []
        scattered_ab = []
        mapping = []
        for i, (l_s, ab_s) in enumerate(zip(stacked_l, stacked_ab)):
            for j, (l, ab) in enumerate(zip(l_s, ab_s)):
                scattered_l.append(l)
                scattered_ab.append(ab)
                mapping.append((i, j))
        return scattered_l, scattered_ab, mapping

    def train(self, data, test_data, logger, config, sample_interval=1):

        # Create folder to save models if needed.
        save_models_path = os.path.join(config.MODEL_DIR, config.TEST_NAME)
        if not os.path.exists(save_models_path):
            os.makedirs(save_models_path)

        # total number of batches in one epoch
        total_batch = int(data.size / data.batch_size)
        print(f'batch_size={data.batch_size} * total_batch={total_batch}')

        for epoch in range(config.NUM_EPOCHS):
            for batch in range(total_batch):
                train_batch = data.generate_batch()
                train_l, train_ab, mapping = self.scatter(train_batch.instances.l, train_batch.instances.ab)

                # Real, Fake and Dummy for Discriminator
                positive_y = np.ones((len(train_l), 1), dtype=np.float32)
                negative_y = -positive_y
                dummy_y = np.zeros((len(train_l), 1), dtype=np.float32)

                # train generator
                g_loss = self.combined.train_on_batch([train_l], [train_ab, positive_y])
                # train discriminator
                d_loss = self.discriminator_model.train_on_batch([train_l, train_ab], [positive_y, negative_y, dummy_y])

                # update log files
                write_log(self.callback, self.train_names, g_loss, (epoch * total_batch + batch + 1))
                write_log(self.callback, self.disc_names, d_loss, (epoch * total_batch + batch + 1))

                if batch % 10 == 0:
                    print("[Epoch %d] [Batch %d/%d] [generator loss: %08f] [discriminator loss: %08f]" % (
                        epoch, batch, total_batch, g_loss[0], d_loss[0]))
            # save models after each epoch
            save_path = os.path.join(save_models_path, "instance_combinedEpoch%d.h5" % epoch)
            self.combined.save(save_path)
            save_path = os.path.join(save_models_path, "instance_colorizationEpoch%d.h5" % epoch)
            self.foreground_generator.save(save_path)
            save_path = os.path.join(save_models_path, "instance_discriminatorEpoch%d.h5" % epoch)
            self.foreground_discriminator.save(save_path)

            # sample images after each epoch
            self.sample_images(test_data, epoch, config)

    def sample_images(self, test_data, epoch, config):
        total_batch = int(test_data.size / test_data.batch_size)
        for _ in range(total_batch):
            test_batch = test_data.generate_batch()
            test_l, test_ab, mapping = self.scatter(test_batch.instances.l, test_batch.instances.ab)
            pred_ab = self.foreground_generator.predict(np.asarray(test_l)[:, :, :, 0])

            for i, (k, l) in enumerate(mapping):
                img = test_batch.images.full[k]
                height, width, _ = img.shape
                bbox = test_batch.instances.bbox[k]
                y_box = np.floor(bbox[:2, l] * height).astype(int)
                x_box = np.floor(bbox[2:, l] * width).astype(int)

                instance_original = img[y_box[0]:y_box[1], x_box[0]:x_box[1], :]
                pred_ab_full_size = cv2.resize(deprocess(pred_ab[i]), (x_box[1]-x_box[0], y_box[1]-y_box[0]))
                labimg_ori = instance_original[:, :, :1]
                reconstruct(deprocess(labimg_ori), pred_ab_full_size, f'epoch{epoch}_{test_batch.file_names[k][:-4]}_{l}', config)


                # originalResult = test_batch.images.full[k]
                # height, width, channels = originalResult.shape
                # predictedAB = cv2.resize(deprocess(pred_ab[i]), (width, height))
                # labimg_ori = np.expand_dims(test_batch.images.l[i], axis=2)
                # reconstruct(deprocess(labimg_ori), predictedAB, f'epoch{epoch}_{test_batch.file_names[i][:-4]}', config)


if __name__ == '__main__':
    config = FirstTestConfig('../../../')
    train_data = Data(config.TRAIN_DIR, config)
    test_data = Data(config.TEST_DIR, config)
    with train(train_data.batch_size, config) as logger:
        logger.write(str(datetime.now()) + "\n")

        print("Initializing Model...")
        colorizationModel = InstanceModel(config)
        print("Model Initialized!")

        print("Start training")
        colorizationModel.train(train_data, test_data, logger, config=config)
