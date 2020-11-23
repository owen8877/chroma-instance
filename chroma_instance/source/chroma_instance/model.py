import datetime
import os
from functools import partial

import cv2
import numpy as np
import tensorflow as tf
from keras import Model, applications
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, BatchNormalization, Dense, Flatten, RepeatVector, Reshape, UpSampling2D
from keras.layers import concatenate, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import _Merge
from keras.optimizers import Adam

from chroma_instance import config, dataClass as data

GRADIENT_PENALTY_WEIGHT = 10


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    save_results_path = os.path.join(config.OUT_DIR, config.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(save_results_path, filelist + "_reconstructed.jpg")
    cv2.imwrite(save_path, result)
    return result


def reconstruct_no(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


def wasserstein_loss_dummy(y_true, y_pred):
    return tf.reduce_mean(y_pred)


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def apply(nets, input):
    for i, net in enumerate(nets):
        if i == 0:
            output = net(input)
        else:
            output = net(output)
    return output


def discriminator_model(shape):
    input_ab = Input(shape=(*shape, 2), name='ab_input')
    input_l = Input(shape=(*shape, 1), name='l_input')
    net = concatenate([input_l, input_ab])
    net = Conv2D(64, (4, 4), padding='same', strides=(2, 2))(net)  # 112, 112, 64
    net = LeakyReLU()(net)
    net = Conv2D(128, (4, 4), padding='same', strides=(2, 2))(net)  # 56, 56, 128
    net = LeakyReLU()(net)
    net = Conv2D(256, (4, 4), padding='same', strides=(2, 2))(net)  # 28, 28, 256
    net = LeakyReLU()(net)
    net = Conv2D(512, (4, 4), padding='same', strides=(1, 1))(net)  # 28, 28, 512
    net = LeakyReLU()(net)
    net = Conv2D(1, (4, 4), padding='same', strides=(1, 1))(net)  # 28, 28,1
    return Model([input_ab, input_l], net)


def colorization_model(shape):
    input_img = Input(shape=(*shape, 3))

    # VGG16 without top layers
    VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model_ = Model(VGG_model.input, VGG_model.layers[-6].output)
    vgg_feature = model_(input_img)

    # Global features
    global_features = apply([
        Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),

        Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),
    ], vgg_feature)

    # Global feature pass back to colorization + classification
    global_merge_back = apply([
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        RepeatVector(28 * 28),
        Reshape((28, 28, 256)),
    ], global_features)

    global_feature_class = apply([
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax'),
    ], global_features)

    # Mid-level features
    midlevel_features = apply([
        Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),
    ], vgg_feature)

    # Fusion of (VGG16 -> Mid-level) + (VGG16 -> Global)
    fusion = concatenate([midlevel_features, global_merge_back])

    # Fusion + Colorization
    output = apply([
        Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu'),
        Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        UpSampling2D(size=(2, 2)),
        Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        UpSampling2D(size=(2, 2)),
        Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid'),
        UpSampling2D(size=(2, 2)),
    ], fusion)

    generated = Model(input=input_img, outputs=[output, global_feature_class])

    return generated


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((config.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class ChromaModel():
    def __init__(self):
        img_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE)

        # Creating generator and discriminator
        optimizer = Adam(0.00002, 0.5)
        self.discriminator = discriminator_model(img_shape)
        self.discriminator.compile(loss=wasserstein_loss_dummy, optimizer=optimizer)

        self.colorization_model = colorization_model(img_shape)
        self.colorization_model.compile(loss=['mse', 'kld'], optimizer=optimizer)

        img_L = Input(shape=(*img_shape, 1))
        img_real_ab = Input(shape=(*img_shape, 2))
        img_L_rep3 = Input(shape=(*img_shape, 3))

        self.colorization_model.trainable = False
        img_pred_ab, class_vec = self.colorization_model(img_L_rep3)
        dis_pred_ab = self.discriminator([img_pred_ab, img_L])
        dis_real_ab = self.discriminator([img_real_ab, img_L])

        # Sample the gradient penalty
        img_ab_interp_samples = RandomWeightedAverage()([img_real_ab, img_pred_ab])
        dis_interp_ab = self.discriminator([img_ab_interp_samples, img_L])
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=img_ab_interp_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'

        # Compile D and G as well as combined
        self.discriminator_model = Model(inputs=[img_L, img_real_ab, img_L_rep3],
                                         outputs=[dis_real_ab,
                                                  dis_pred_ab,
                                                  dis_interp_ab])

        self.discriminator_model.compile(optimizer=optimizer,
                                         loss=[wasserstein_loss_dummy,
                                               wasserstein_loss_dummy,
                                               partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])

        self.colorization_model.trainable = True
        self.discriminator.trainable = False
        self.combined = Model(inputs=[img_L_rep3, img_L],
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

    def train(self, data, test_data, log, sample_interval=1):

        # Create folder to save models if needed.
        save_models_path = os.path.join(config.MODEL_DIR, config.TEST_NAME)
        if not os.path.exists(save_models_path):
            os.makedirs(save_models_path)

        # Load VGG network
        VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)

        # Real, Fake and Dummy for Discriminator
        positive_y = np.ones((config.BATCH_SIZE, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((config.BATCH_SIZE, 1), dtype=np.float32)

        # total number of batches in one epoch
        total_batch = int(data.size / config.BATCH_SIZE)

        for epoch in range(config.NUM_EPOCHS):
            for batch in range(total_batch):
                # new batch
                trainL, trainAB, _, original, l_img_oritList = data.generate_batch()
                l_3 = np.tile(trainL, [1, 1, 1, 3])

                # GT vgg
                predictVGG = VGG_modelF.predict(l_3)

                # train generator
                g_loss = self.combined.train_on_batch([l_3, trainL],
                                                      [trainAB, predictVGG, positive_y])
                # train discriminator
                d_loss = self.discriminator_model.train_on_batch([trainL, trainAB, l_3],
                                                                 [positive_y, negative_y, dummy_y])

                # update log files
                write_log(self.callback, self.train_names, g_loss, (epoch * total_batch + batch + 1))
                write_log(self.callback, self.disc_names, d_loss, (epoch * total_batch + batch + 1))

                if (batch) % 1000 == 0:
                    print("[Epoch %d] [Batch %d/%d] [generator loss: %08f] [discriminator loss: %08f]" % (
                        epoch, batch, total_batch, g_loss[0], d_loss[0]))
            # save models after each epoch
            save_path = os.path.join(save_models_path, "my_model_combinedEpoch%d.h5" % epoch)
            self.combined.save(save_path)
            save_path = os.path.join(save_models_path, "my_model_colorizationEpoch%d.h5" % epoch)
            self.colorization_model.save(save_path)
            save_path = os.path.join(save_models_path, "my_model_discriminatorEpoch%d.h5" % epoch)
            self.discriminator.save(save_path)

            # sample images after each epoch
            self.sample_images(test_data, epoch)

    def sample_images(self, test_data, epoch):
        total_batch = int(test_data.size / config.BATCH_SIZE)
        for _ in range(total_batch):
            # load test data
            testL, _, filelist, original, labimg_oritList = test_data.generate_batch()

            # predict AB channels
            predAB, _ = self.colorization_model.predict(np.tile(testL, [1, 1, 1, 3]))

            # print results
            for i in range(config.BATCH_SIZE):
                originalResult = original[i]
                height, width, channels = originalResult.shape
                predictedAB = cv2.resize(deprocess(predAB[i]), (width, height))
                labimg_ori = np.expand_dims(labimg_oritList[i], axis=2)
                predResult = reconstruct(deprocess(labimg_ori), predictedAB,
                                         "epoch" + str(epoch) + "_" + filelist[i][:-5])


def train():
    # Create log folder if needed.
    log_path = os.path.join(config.LOG_DIR, config.TEST_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(
            config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".txt"), "w") as log:
        log.write(str(datetime.datetime.now()) + "\n")

        print('load training data from ' + config.TRAIN_DIR)
        train_data = data.DATA(config.TRAIN_DIR)
        test_data = data.DATA(config.TEST_DIR)
        assert config.BATCH_SIZE <= train_data.size, "The batch size should be smaller or equal to the number of training images --> modify it in config.py"
        print("Train data loaded")

        print("Initializing Model...")
        colorizationModel = ChromaModel()
        print("Model Initialized!")

        print("Start training")
        colorizationModel.train(train_data, test_data, log)


if __name__ == '__main__':
    train()