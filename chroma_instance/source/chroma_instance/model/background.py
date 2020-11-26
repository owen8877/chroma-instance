import os
from functools import partial

import cv2
import numpy as np
from keras import Model, applications
from keras.callbacks import TensorBoard
from keras.layers import Input
from keras.optimizers import Adam

from chroma_instance.model.basic import discriminator_network, bg_colorization_network, RandomWeightedAverage, \
    wasserstein_loss_dummy, gradient_penalty_loss
from chroma_instance.util import write_log, deprocess, reconstruct

GRADIENT_PENALTY_WEIGHT = 10


class BackgroundModel:
    def __init__(self, config):
        img_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE)

        # Creating generator and discriminator
        optimizer = Adam(0.00002, 0.5)

        self.background_discriminator = discriminator_network(img_shape)
        self.background_discriminator.compile(loss=wasserstein_loss_dummy, optimizer=optimizer)
        self.background_generator = bg_colorization_network(img_shape)
        self.background_generator.compile(loss=['mse', 'kld'], optimizer=optimizer)

        img_L = Input(shape=(*img_shape, 1))
        img_real_ab = Input(shape=(*img_shape, 2))

        self.background_generator.trainable = False
        img_pred_ab, class_vec = self.background_generator(img_L)
        dis_pred_ab = self.background_discriminator([img_pred_ab, img_L])
        dis_real_ab = self.background_discriminator([img_real_ab, img_L])

        # Sample the gradient penalty
        img_ab_interp_samples = RandomWeightedAverage()([img_real_ab, img_pred_ab])
        dis_interp_ab = self.background_discriminator([img_ab_interp_samples, img_L])
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=img_ab_interp_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'

        # Compile D and G as well as combined
        self.discriminator_model = Model(inputs=[img_L, img_real_ab],
                                         outputs=[dis_real_ab,
                                                  dis_pred_ab,
                                                  dis_interp_ab])

        self.discriminator_model.compile(optimizer=optimizer,
                                         loss=[wasserstein_loss_dummy,
                                               wasserstein_loss_dummy,
                                               partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])

        self.background_generator.trainable = True
        self.background_discriminator.trainable = False
        self.combined = Model(inputs=[img_L],
                              outputs=[img_pred_ab, class_vec, dis_pred_ab])
        self.combined.compile(loss=['mse', 'kld', wasserstein_loss_dummy],
                              loss_weights=[1.0, 0.003, -0.1],
                              optimizer=optimizer)

        # Monitor stuff
        self.log_path = os.path.join(config.LOG_DIR, config.TEST_NAME)
        self.callback = TensorBoard(self.log_path)
        self.callback.set_model(self.combined)
        self.train_names = ['loss', 'mse_loss', 'kullback_loss', 'wasserstein_loss']
        self.disc_names = ['disc_loss', 'disc_valid', 'disc_fake', 'disc_gp']

        self.test_loss_array = []
        self.g_loss_array = []

    def train(self, data, test_data, log, config, sample_interval=1):

        # Create folder to save models if needed.
        save_models_path = os.path.join(config.MODEL_DIR, config.TEST_NAME)
        if not os.path.exists(save_models_path):
            os.makedirs(save_models_path)

        # Load VGG network
        VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)

        # Real, Fake and Dummy for Discriminator
        positive_y = np.ones((data.batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((data.batch_size, 1), dtype=np.float32)

        # total number of batches in one epoch
        total_batch = int(data.size / data.batch_size)
        print(f'batch_size={data.batch_size} * total_batch={total_batch}')

        for epoch in range(config.NUM_EPOCHS):
            for batch in range(total_batch):
                # new batch
                train_l, train_ab, _, _, _, train_bbox = data.generate_batch()
                train_l3 = np.tile(train_l, [1, 1, 1, 3])

                # GT vgg
                predictVGG = VGG_modelF.predict(train_l3)

                # train generator
                g_loss = self.combined.train_on_batch([train_l], [train_ab, predictVGG, positive_y])
                # train discriminator
                d_loss = self.discriminator_model.train_on_batch([train_l, train_ab],
                                                                 [positive_y, negative_y, dummy_y])

                # update log files
                write_log(self.callback, self.train_names, g_loss, (epoch * total_batch + batch + 1))
                write_log(self.callback, self.disc_names, d_loss, (epoch * total_batch + batch + 1))

                if batch % 10 == 0:
                    print("[Epoch %d] [Batch %d/%d] [generator loss: %08f] [discriminator loss: %08f]" % (
                        epoch, batch, total_batch, g_loss[0], d_loss[0]))
            # save models after each epoch
            save_path = os.path.join(save_models_path, "background_combinedEpoch%d.h5" % epoch)
            self.combined.save(save_path)
            save_path = os.path.join(save_models_path, "background_colorizationEpoch%d.h5" % epoch)
            self.background_generator.save(save_path)
            save_path = os.path.join(save_models_path, "background_discriminatorEpoch%d.h5" % epoch)
            self.background_discriminator.save(save_path)

            # sample images after each epoch
            self.sample_images(test_data, epoch, config)

    def sample_images(self, test_data, epoch, config):
        total_batch = int(test_data.size / test_data.batch_size)
        for _ in range(total_batch):
            # load test data
            testL, _, filelist, original, labimg_oritList = test_data.generate_batch()

            # predict AB channels
            predAB, _ = self.background_generator.predict(testL)

            # print results
            for i in range(test_data.batch_size):
                originalResult = original[i]
                height, width, channels = originalResult.shape
                predictedAB = cv2.resize(deprocess(predAB[i]), (width, height))
                labimg_ori = np.expand_dims(labimg_oritList[i], axis=2)
                reconstruct(deprocess(labimg_ori), predictedAB, f'epoch{epoch}_{filelist[i][:-5]}', config)
