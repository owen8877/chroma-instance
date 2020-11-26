import os
from functools import partial

import cv2
import numpy as np
import tensorflow as tf
from keras import Model, applications, Sequential
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Conv2D
from keras.layers import Input, Layer
from keras.layers.advanced_activations import Softmax
from keras.optimizers import Adam

from chroma_instance.data.generator import Data
from chroma_instance.model.basic import discriminator_network, RandomWeightedAverage, \
    fusion_network, instance_colorization_network
from chroma_instance.util import write_log, deprocess, reconstruct

GRADIENT_PENALTY_WEIGHT = 10


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


def wasserstein_loss_dummy(y_true, y_pred):
    return tf.reduce_mean(y_pred)


class WeightGenerator(Layer):
    def __init__(self, units, box_info):
        super(WeightGenerator, self).__init__()
        self.box_info = box_info

        instance_conv = Sequential()
        instance_conv.add(Conv2D(units, 3, padding='same', strides=1, activation='relu'))
        instance_conv.add(Conv2D(units, 3, padding='same', strides=1, activation='relu'))
        instance_conv.add(Conv2D(1, 3, padding='same', strides=1, activation='relu'))
        self.instance_conv = instance_conv

        simple_bg_conv = Sequential()
        simple_bg_conv.add(Conv2D(units, 3, padding='same', strides=1, activation='relu'))
        simple_bg_conv.add(Conv2D(units, 3, padding='same', strides=1, activation='relu'))
        simple_bg_conv.add(Conv2D(1, 3, padding='same', strides=1, activation='relu'))
        self.simple_bg_conv = simple_bg_conv

        self.normalize = Softmax(1)

    def resize_and_pad(self, feauture_maps, info_array):
        # TODO: check bounding box definition
        feauture_maps = tf.image.resize(feauture_maps, (info_array[5], info_array[
            4]))  # torch.nn.functional.interpolate(feauture_maps, size=(info_array[5], info_array[4]), mode='bilinear')
        feauture_maps = tf.pad(feauture_maps, tf.constant([[0, 0], [info_array[2], info_array[3]], [info_array[0],
                                                                                                    info_array[
                                                                                                        1]]]))  # torch.nn.functional.pad(feauture_maps, (info_array[0], info_array[1], info_array[2], info_array[3]), "constant", 0)
        return feauture_maps

    def call(self, inputs, **kwargs):
        instance_feature, bg_feature = inputs

        mask_list = []
        feature_map_list = []
        mask_sum_for_pred = tf.zeros_like(bg_feature)[:1, :1]  # torch.zeros_like(bg_feature)[:1, :1]
        for i in range(instance_feature.shape[0]):
            box_info = self.box_info[i]
            tmp_crop = tf.expand_dims(instance_feature[i], axis=0)  # torch.unsqueeze(instance_feature[i], 0)
            conv_tmp_crop = self.instance_conv(tmp_crop)
            pred_mask = self.resize_and_pad(conv_tmp_crop, box_info)

            tmp_crop = self.resize_and_pad(tmp_crop, box_info)

            mask = tf.zeros_like(bg_feature)[:1, :1]  # torch.zeros_like(bg_feature)[:1, :1]
            # TODO: check bounding box definition
            left = box_info[2]
            right = box_info[2] + box_info[5]
            bottom = box_info[0]
            top = box_info[0] + box_info[4]
            mask[0, 0, left:right, bottom:top] = 1.0
            # TODO: check if cast is necessary
            # device = mask.device
            # mask = mask.type(torch.FloatTensor).to(device)

            mask_sum_for_pred = tf.clip_by_value(mask_sum_for_pred + mask, 0.0,
                                                 1.0)  # torch.clamp(mask_sum_for_pred + mask, 0.0, 1.0)

            mask_list.append(pred_mask)
            feature_map_list.append(tmp_crop)

        pred_bg_mask = self.simple_bg_conv(bg_feature)
        mask_list.append(pred_bg_mask + (1 - mask_sum_for_pred) * 100000.0)
        mask_list = self.normalize(tf.concat(mask_list, 1))  # self.normalize(torch.cat(mask_list, 1))

        mask_list_maskout = tf.identity(mask_list)  # mask_list.clone()

        instance_mask = tf.clip_by_value(
            tf.reduce_sum(mask_list_maskout[:, :instance_feature.shape[0]], axis=1, keepdim=True), 0.0,
            1.0)  # torch.clamp(torch.sum(mask_list_maskout[:, :instance_feature.shape[0]], 1, keepdim=True), 0.0, 1.0)

        feature_map_list.append(bg_feature)
        feature_map_list = tf.concat(feature_map_list, 0)  # torch.cat(feature_map_list, 0)
        # TODO: check dimension
        # TODO: no corresponding API for .contiguous()
        mask_list_maskout = tf.keras.backend.permute_dimensions(mask_list_maskout, (
            1, 0, 2, 3))  # mask_list_maskout.permute(1, 0, 2, 3).contiguous()
        # TODO: do we need tf.matmul()?
        out = feature_map_list * mask_list_maskout
        out = tf.reduce_sum(out, 0, keepdim=True)  # out = torch.sum(out, 0, keepdim=True)
        return out  # , instance_mask, torch.clamp(mask_list, 0.0, 1.0)


class FusionModel:
    def __init__(self, config):
        img_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE)

        # Creating generator and discriminator
        optimizer = Adam(0.00002, 0.5)

        self.foreground_generator = instance_colorization_network(img_shape)
        self.foreground_generator.compile(loss=['mse', 'kld'], optimizer=optimizer)

        self.fusion_discriminator = discriminator_network(img_shape)
        self.fusion_discriminator.compile(loss=wasserstein_loss_dummy, optimizer=optimizer)
        self.fusion_generator = fusion_network(img_shape)
        self.fusion_generator.compile(loss=['mse', 'kld'], optimizer=optimizer)

        # Fg=instance prediction
        fg_img_l = Input(shape=(*img_shape, 1, None))

        self.foreground_generator.trainable = False
        fg_img_pred_ab, fg_class_vec, fg_feature = self.foreground_generator(fg_img_l)

        # Fusion prediction
        fusion_img_l = Input(shape=(*img_shape, 1))
        fusion_img_real_ab = Input(shape=(*img_shape, 2))
        # TODO: check if None can be used for a dimension placeholder?
        fg_bbox = Input(shape=(4, None))
        fg_mask = Input(shape=(*img_shape, None))

        self.fusion_generator.trainable = False
        fusion_img_pred_ab, fusion_class_vec = self.fusion_generator(fusion_img_l, fg_feature, fg_bbox, fg_mask)

        dis_pred_ab = self.fusion_discriminator([fusion_img_pred_ab, fusion_img_l])
        dis_real_ab = self.fusion_discriminator([fusion_img_real_ab, fusion_img_l])

        # Sample the gradient penalty
        img_ab_interp_samples = RandomWeightedAverage()([fusion_img_pred_ab, fusion_img_real_ab])
        dis_interp_ab = self.fusion_discriminator([img_ab_interp_samples, fusion_img_l])
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=img_ab_interp_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'

        # Compile D and G as well as combined
        self.discriminator_model = Model(
            inputs=[fusion_img_l, fusion_img_real_ab, fg_bbox, fg_mask],
            outputs=[dis_real_ab,
                     dis_pred_ab,
                     dis_interp_ab])

        self.discriminator_model.compile(optimizer=optimizer,
                                         loss=[wasserstein_loss_dummy,
                                               wasserstein_loss_dummy,
                                               partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])

        self.fusion_generator.trainable = True
        self.fusion_discriminator.trainable = False
        self.combined = Model(inputs=[fusion_img_l, fg_img_l, fg_bbox, fg_mask],
                              outputs=[fusion_img_pred_ab, fusion_class_vec, dis_pred_ab])
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

    def train(self, data: Data, test_data, log, config, sample_interval=1):

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
                train_batch = data.generate_batch()
                resized_l = train_batch.resized_images.l
                resized_ab = train_batch.resized_images.ab
                resized_l3 = np.tile(resized_l, [1, 1, 1, 3])

                # GT vgg
                predictVGG = VGG_modelF.predict(resized_l3)

                # train generator
                g_loss = self.combined.train_on_batch([resized_l], [resized_ab, predictVGG, positive_y])
                # train discriminator
                d_loss = self.discriminator_model.train_on_batch(
                    [resized_l, resized_ab, train_batch.instances.l, train_batch.instances.bbox,
                     train_batch.instances.mask],
                    [positive_y, negative_y, dummy_y])

                # update log files
                write_log(self.callback, self.train_names, g_loss, (epoch * total_batch + batch + 1))
                write_log(self.callback, self.disc_names, d_loss, (epoch * total_batch + batch + 1))

                if batch % 10 == 0:
                    print("[Epoch %d] [Batch %d/%d] [generator loss: %08f] [discriminator loss: %08f]" % (
                        epoch, batch, total_batch, g_loss[0], d_loss[0]))
            # save models after each epoch
            save_path = os.path.join(save_models_path, "fusion_combinedEpoch%d.h5" % epoch)
            self.combined.save(save_path)
            save_path = os.path.join(save_models_path, "fusion_colorizationEpoch%d.h5" % epoch)
            self.fusion_generator.save(save_path)
            save_path = os.path.join(save_models_path, "fusion_instance_colorizationEpoch%d.h5" % epoch)
            self.foreground_generator.save(save_path)
            save_path = os.path.join(save_models_path, "fusion_discriminatorEpoch%d.h5" % epoch)
            self.fusion_discriminator.save(save_path)

            # sample images after each epoch
            self.sample_images(test_data, epoch, config)

    def sample_images(self, test_data: Data, epoch, config):
        total_batch = int(test_data.size / test_data.batch_size)
        for _ in range(total_batch):
            # load test data
            test_batch = test_data.generate_batch()

            # predict AB channels
            _, _, fg_feature = self.foreground_generator.predict(test_batch.instances.l)
            fusion_pred_ab, _ = self.fusion_generator.predict(test_batch.resized_images.l, test_batch.instances.l,
                                                              test_batch.instances.bbox, test_batch.instances.mask)

            # print results
            for i in range(test_data.batch_size):
                originalResult = test_batch.images.full[i]
                height, width, channels = originalResult.shape
                predictedAB = cv2.resize(deprocess(fusion_pred_ab[i]), (width, height))
                labimg_ori = np.expand_dims(test_batch.images.l[i], axis=2)
                reconstruct(deprocess(labimg_ori), predictedAB, f'epoch{epoch}_{test_batch.file_names[i][:-5]}', config)
