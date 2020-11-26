import os
from datetime import datetime
from functools import partial

import cv2
import numpy as np
import tensorflow as tf
from keras import Model, applications, Sequential
from keras.callbacks import TensorBoard
from keras.layers import Input, BatchNormalization, Dense, Flatten, RepeatVector, Reshape, UpSampling2D
from keras.layers import Layer
from keras.layers import concatenate, Conv2D
from keras.layers.advanced_activations import Softmax
from keras.layers.merge import Concatenate
from keras.optimizers import Adam

from chroma_instance.config import MAX_INSTANCES
from chroma_instance.config.FirstTest import FirstTestConfig
from chroma_instance.data.generator import Data
from chroma_instance.model.basic import discriminator_network, RandomWeightedAverage, wasserstein_loss_dummy, \
    gradient_penalty_loss
from chroma_instance.util import write_log, deprocess, reconstruct, train

GRADIENT_PENALTY_WEIGHT = 10


class WeightGenerator(Layer):
    def __init__(self, units):
        super(WeightGenerator, self).__init__()

        instance_conv = Sequential()
        instance_conv.add(Conv2D(units, 3, padding='same', strides=1, activation='relu'))
        # instance_conv.add(Conv2D(units, 3, padding='same', strides=1, activation='relu'))
        instance_conv.add(Conv2D(1, 3, padding='same', strides=1, activation='relu'))
        self.instance_conv = instance_conv

        simple_bg_conv = Sequential()
        simple_bg_conv.add(Conv2D(units, 3, padding='same', strides=1, activation='relu'))
        # simple_bg_conv.add(Conv2D(units, 3, padding='same', strides=1, activation='relu'))
        simple_bg_conv.add(Conv2D(1, 3, padding='same', strides=1, activation='relu'))
        self.simple_bg_conv = simple_bg_conv

        self.softmax_beta = 1
        self.batch_n = 10

    def resize_and_pad(self, feauture_maps, info_array, bg_size, pad_3=False):
        y_size = int(bg_size[0] * (info_array[1] - info_array[0]))
        x_size = int(bg_size[1] * (info_array[3] - info_array[2]))
        feauture_maps = tf.image.resize(feauture_maps, (y_size, x_size))
        y_front_pad = int(bg_size[0] * info_array[0])
        x_front_pad = int(bg_size[1] * info_array[2])
        padding = [
            [y_front_pad, bg_size[0] - y_front_pad - y_size],
            [x_front_pad, bg_size[1] - x_front_pad - x_size]
        ]
        if pad_3:
            padding.append([0, 0])
        feauture_maps = tf.pad(feauture_maps, tf.constant(padding))
        return feauture_maps

    def call(self, inputs, **kwargs):
        # TODO: How to use mrcnn_mask?
        instance_feature, bg_feature, bbox, mrcnn_mask, object_n = inputs
        bg_size = bg_feature.shape[1:3]

        output = []
        for i in range(self.batch_n):
            resized_sub_weights = []
            resized_masks = []
            for j in range(object_n(i)):
                box_info = bbox[i, :, j]
                instance_sub_feature = instance_feature[i, :, :, :, j]
                instance_weight = self.instance_conv(instance_sub_feature)
                resized_sub_weight = self.resize_and_pad(instance_weight, box_info, bg_size)

                instance_mask = mrcnn_mask[i, :, :, j]
                resized_mask = self.resize_and_pad(instance_weight, box_info, bg_size)
                resized_mask = tf.where(resized_mask > 0, 1.0, 0.0)

                resized_sub_weights.append(resized_sub_weight)
                resized_masks.append(resized_mask)
            bg_sub_feature = bg_feature[i, :, :, :]
            bg_weight = self.simple_bg_conv(bg_sub_feature)

            exp_weight = []
            for f, m in zip(resized_sub_weights, resized_masks):
                exp_weight.append(tf.exp(f) * m)
            exp_weight.append(tf.exp(bg_weight))

            exp_weight_sum = sum(exp_weight)
            softmax_weight = []
            for w in exp_weight:
                softmax_weight.append(w / exp_weight_sum)

            composed = softmax_weight[-1] * bg_sub_feature
            for j in range(object_n(i)):
                box_info = bbox[i, :, j]
                instance_sub_feature = instance_feature[i, :, :, :, j]
                resized_sub_feature = self.resize_and_pad(instance_sub_feature, box_info, bg_size)
                composed += resized_sub_feature * softmax_weight[j]

            output.append(composed)


        #         tmp_crop = tf.expand_dims(instance_feature[i], axis=0)  # torch.unsqueeze(instance_feature[i], 0)
        #         conv_tmp_crop = self.instance_conv(tmp_crop)
        #         pred_mask = self.resize_and_pad(conv_tmp_crop, box_info)
        #
        #         tmp_crop = self.resize_and_pad(tmp_crop, box_info)
        #
        #         mask = tf.zeros_like(bg_feature)[:1, :1]  # torch.zeros_like(bg_feature)[:1, :1]
        #         # TODO: check bounding box definition
        #         left = box_info[2]
        #         right = box_info[2] + box_info[5]
        #         bottom = box_info[0]
        #         top = box_info[0] + box_info[4]
        #         mask[0, 0, left:right, bottom:top] = 1.0
        #         # TODO: check if cast is necessary
        #         # device = mask.device
        #         # mask = mask.type(torch.FloatTensor).to(device)
        #
        #         mask_sum_for_pred = tf.clip_by_value(mask_sum_for_pred + mask, 0.0,
        #                                              1.0)  # torch.clamp(mask_sum_for_pred + mask, 0.0, 1.0)
        #
        #         mask_list.append(pred_mask)
        #         feature_map_list.append(tmp_crop)
        #
        # pred_bg_mask = self.simple_bg_conv(bg_feature)
        # mask_list.append(pred_bg_mask + (1 - mask_sum_for_pred) * 100000.0)
        # mask_list = self.normalize(tf.concat(mask_list, 1))  # self.normalize(torch.cat(mask_list, 1))
        #
        # mask_list_maskout = tf.identity(mask_list)  # mask_list.clone()
        #
        # instance_mask = tf.clip_by_value(
        #     tf.reduce_sum(mask_list_maskout[:, :instance_feature.shape[0]], axis=1, keepdim=True), 0.0,
        #     1.0)  # torch.clamp(torch.sum(mask_list_maskout[:, :instance_feature.shape[0]], 1, keepdim=True), 0.0, 1.0)
        #
        # feature_map_list.append(bg_feature)
        # feature_map_list = tf.concat(feature_map_list, 0)  # torch.cat(feature_map_list, 0)
        # # TODO: check dimension
        # # TODO: no corresponding API for .contiguous()
        # mask_list_maskout = tf.keras.backend.permute_dimensions(mask_list_maskout, (
        #     1, 0, 2, 3))  # mask_list_maskout.permute(1, 0, 2, 3).contiguous()
        # # TODO: do we need tf.matmul()?
        # out = feature_map_list * mask_list_maskout
        # out = tf.reduce_sum(out, 0, keepdim=True)  # out = torch.sum(out, 0, keepdim=True)
        return output  # , instance_mask, torch.clamp(mask_list, 0.0, 1.0)


def chroma_color_with_label(shape):
    input_img = Input(shape=shape)
    input_img_1 = Reshape((*shape, 1))(input_img)
    input_img_3 = Concatenate(axis=3)([input_img_1, input_img_1, input_img_1])

    # VGG16 without top layers
    VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model_3 = Model(VGG_model.input, VGG_model.layers[-6].output, name='fg_model_3')(input_img_3)

    # Global features
    conv2d_6 = Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu', name='fg_conv2d_6')(model_3)
    batch_normalization_1 = BatchNormalization(name='fg_batch_normalization_1')(conv2d_6)
    conv2d_7 = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_7')(
        batch_normalization_1)
    batch_normalization_2 = BatchNormalization(name='fg_batch_normalization_2')(conv2d_7)

    conv2d_8 = Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu', name='fg_conv2d_8')(
        batch_normalization_2)
    batch_normalization_3 = BatchNormalization(name='fg_batch_normalization_3')(conv2d_8)
    conv2d_9 = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_9')(
        batch_normalization_3)
    batch_normalization_4 = BatchNormalization(name='fg_batch_normalization_4')(conv2d_9)

    # Global feature pass back to colorization + classification
    flatten_1 = Flatten(name='fg_flatten_1')(batch_normalization_4)
    dense_1 = Dense(1024, activation='relu', name='fg_dense_1')(flatten_1)
    dense_2 = Dense(512, activation='relu', name='fg_dense_2')(dense_1)
    dense_3 = Dense(256, activation='relu', name='fg_dense_3')(dense_2)
    repeat_vector_1 = RepeatVector(28 * 28, name='fg_repeat_vector_1')(dense_3)
    reshape_1 = Reshape((28, 28, 256), name='fg_reshape_1')(repeat_vector_1)

    # Mid-level features
    conv2d_10 = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_10')(model_3)
    batch_normalization_5 = BatchNormalization(name='fg_batch_normalization_5')(conv2d_10)
    conv2d_11 = Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_11')(
        batch_normalization_5)
    batch_normalization_6 = BatchNormalization(name='fg_batch_normalization_6')(conv2d_11)

    # Fusion of (VGG16 -> Mid-level) + (VGG16 -> Global) + Colorization
    concatenate_2 = concatenate([batch_normalization_6, reshape_1], name='fg_concatenate_2')

    conv2d_12 = Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_12')(
        concatenate_2)
    conv2d_13 = Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_13')(conv2d_12)
    up_sampling2d_1 = UpSampling2D(size=(2, 2), name='fg_up_sampling2d_1')(conv2d_13)

    conv2d_14 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_14')(
        up_sampling2d_1)
    conv2d_15 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_15')(conv2d_14)
    up_sampling2d_2 = UpSampling2D(size=(2, 2), name='fg_up_sampling2d_2')(conv2d_15)

    conv2d_16 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_16')(
        up_sampling2d_2)
    conv2d_17 = Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid', name='fg_conv2d_17')(conv2d_16)
    up_sampling2d_3 = UpSampling2D(size=(2, 2), name='fg_up_sampling2d_3')(conv2d_17)

    generated = Model(input=input_img, outputs=[model_3, conv2d_11, conv2d_13, conv2d_15, conv2d_17])

    return generated


class InstanceColorModel(Model):
    def __init__(self, shape):
        super(InstanceColorModel, self).__init__()
        self.model = chroma_color_with_label(shape)

    def call(self, inputs, mask=None):
        input, object_n = inputs

        output_n = 0
        batch_n = input.shape[0]
        all_features = [[] for _ in range(batch_n)]
        for i in range(batch_n):
            for j in range(object_n[i]):
                example = self.model(input[i, :, :, : j])
                all_features[i].append(example)
                output_n = len(example)

        if output_n == 0:
            return [None for _ in range(5)]

        collection = []
        for l in range(output_n):
            collection.append(tf.zeros((batch_n, example.shape[1], example.shape[2], example.shape[3], 4)))
            for i in range(batch_n):
                for j in range(object_n[i]):
                    collection[l][i, :, :, :, j] = all_features[i][j]
        return collection


def fusion_network(shape):
    input_img = Input(shape=shape)
    input_img_1 = Reshape((*shape, 1))(input_img)
    input_img_3 = Concatenate(axis=3)([input_img_1, input_img_1, input_img_1])

    bbox = Input(shape=(4, MAX_INSTANCES))
    mask = Input(shape=(*shape, MAX_INSTANCES))
    object_n = Input(shape=(1,))

    # VGG16 without top layers
    VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_model_3_pre = Model(VGG_model.input, VGG_model.layers[-6].output, name='model_3')(input_img_3)
    fg_model_3 = Input(shape=(*vgg_model_3_pre.shape[1:], MAX_INSTANCES), name='fg_model_3')  # <-
    vgg_model_3 = WeightGenerator(64)([fg_model_3, vgg_model_3_pre, bbox, mask, object_n])  # <-

    # Global features
    conv2d_6 = Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu', name='conv2d_6')(vgg_model_3)
    batch_normalization_1 = BatchNormalization(name='batch_normalization_1')(conv2d_6)
    conv2d_7 = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_7')(
        batch_normalization_1)
    batch_normalization_2 = BatchNormalization(name='batch_normalization_2')(conv2d_7)

    conv2d_8 = Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu', name='conv2d_8')(
        batch_normalization_2)
    batch_normalization_3 = BatchNormalization(name='batch_normalization_3')(conv2d_8)
    conv2d_9 = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_9')(
        batch_normalization_3)
    batch_normalization_4 = BatchNormalization(name='batch_normalization_4')(conv2d_9)

    # Classification
    flatten_2 = Flatten(name='flatten_2')(batch_normalization_4)
    dense_4 = Dense(4096, activation='relu', name='dense_4')(flatten_2)
    dense_5 = Dense(4096, activation='relu', name='dense_5')(dense_4)
    dense_6 = Dense(1000, activation='softmax', name='dense_6')(dense_5)

    # Global feature pass back to colorization + classification
    flatten_1 = Flatten(name='flatten_1')(batch_normalization_4)
    dense_1 = Dense(1024, activation='relu', name='dense_1')(flatten_1)
    dense_2 = Dense(512, activation='relu', name='dense_2')(dense_1)
    dense_3 = Dense(256, activation='relu', name='dense_3')(dense_2)
    repeat_vector_1 = RepeatVector(28 * 28, name='repeat_vector_1')(dense_3)
    reshape_1 = Reshape((28, 28, 256), name='reshape_1')(repeat_vector_1)

    # Mid-level features
    conv2d_10 = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_10')(vgg_model_3)
    batch_normalization_5 = BatchNormalization(name='batch_normalization_5')(conv2d_10)
    conv2d_11_pre = Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_11')(
        batch_normalization_5)
    fg_conv2d_11 = Input(shape=(*conv2d_11_pre.shape[1:], MAX_INSTANCES), name='fg_conv2d_11')  # <-
    conv2d_11 = WeightGenerator(32)([fg_conv2d_11, conv2d_11_pre, bbox, mask, object_n])  # <-
    batch_normalization_6 = BatchNormalization(name='batch_normalization_6')(conv2d_11)

    # Fusion of (VGG16 -> Mid-level) + (VGG16 -> Global) + Colorization
    concatenate_2 = concatenate([batch_normalization_6, reshape_1], name='concatenate_2')

    conv2d_12 = Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv2d_12')(concatenate_2)
    conv2d_13_pre = Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_13')(conv2d_12)
    fg_conv2d_13 = Input(shape=(*conv2d_13_pre.shape[1:], MAX_INSTANCES), name='fg_conv2d_13')  # <-
    conv2d_13 = WeightGenerator(16)([fg_conv2d_13, conv2d_13_pre, bbox, mask, object_n])  # <-
    up_sampling2d_1 = UpSampling2D(size=(2, 2), name='up_sampling2d_1')(conv2d_13)

    conv2d_14 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_12')(up_sampling2d_1)
    conv2d_15_pre = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_13')(conv2d_14)
    fg_conv2d_15 = Input(shape=(*conv2d_13_pre.shape[1:], MAX_INSTANCES), name='fg_conv2d_15')  # <-
    conv2d_15 = WeightGenerator(16)([fg_conv2d_15, conv2d_15_pre, bbox, mask, object_n])  # <-
    up_sampling2d_2 = UpSampling2D(size=(2, 2), name='up_sampling2d_2')(conv2d_15)

    conv2d_16 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_12')(up_sampling2d_2)
    conv2d_17_pre = Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid', name='conv2d_13')(conv2d_16)
    fg_conv2d_17 = Input(shape=(*conv2d_17_pre.shape[1:], MAX_INSTANCES), name='fg_conv2d_17')  # <-
    conv2d_17 = WeightGenerator(16)([fg_conv2d_17, conv2d_17_pre, bbox, mask, object_n])  # <-
    up_sampling2d_3 = UpSampling2D(size=(2, 2), name='up_sampling2d_3')(conv2d_17)

    return Model(inputs=[input_img, fg_model_3, fg_conv2d_11, fg_conv2d_13, fg_conv2d_15, fg_conv2d_17, bbox, mask],
                 outputs=[up_sampling2d_3, dense_6])


class FusionModel:
    def __init__(self, config):
        img_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE)

        # Creating generator and discriminator
        optimizer = Adam(0.00002, 0.5)

        self.foreground_generator = InstanceColorModel(img_shape)

        self.fusion_discriminator = discriminator_network(img_shape)
        self.fusion_discriminator.compile(loss=wasserstein_loss_dummy, optimizer=optimizer)
        self.fusion_generator = fusion_network(img_shape)
        self.fusion_generator.compile(loss=['mse', 'kld'], optimizer=optimizer)

        # Fg=instance prediction
        fg_img_l = Input(shape=(*img_shape, 1, None))
        fg_object_n = Input(shape=(1,))

        self.foreground_generator.trainable = False
        fg_model_3, fg_conv2d_11, fg_conv2d_13, fg_conv2d_15, fg_conv2d_17 = self.foreground_generator(
            [fg_img_l, fg_object_n])

        # Fusion prediction
        fusion_img_l = Input(shape=(*img_shape, 1))
        fusion_img_real_ab = Input(shape=(*img_shape, 2))
        # TODO: check if None can be used for a dimension placeholder?
        fg_bbox = Input(shape=(4, None))
        fg_mask = Input(shape=(*img_shape, None))

        self.fusion_generator.trainable = False
        fusion_img_pred_ab, fusion_class_vec = self.fusion_generator([fusion_img_l, fg_model_3, fg_conv2d_11,
                                                                      fg_conv2d_13, fg_conv2d_15, fg_conv2d_17, fg_bbox,
                                                                      fg_mask, fg_object_n])

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
            inputs=[fusion_img_l, fusion_img_real_ab, fg_bbox, fg_mask, fg_object_n],
            outputs=[dis_real_ab,
                     dis_pred_ab,
                     dis_interp_ab])

        self.discriminator_model.compile(optimizer=optimizer,
                                         loss=[wasserstein_loss_dummy,
                                               wasserstein_loss_dummy,
                                               partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])

        self.fusion_generator.trainable = True
        self.fusion_discriminator.trainable = False
        self.combined = Model(inputs=[fusion_img_l, fg_img_l, fg_bbox, fg_mask, fg_object_n],
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


if __name__ == '__main__':
    config = FirstTestConfig('../../../')
    train_data = Data(config.TRAIN_DIR, config)
    test_data = Data(config.TEST_DIR, config)
    with train(train_data.batch_size, config) as logger:
        logger.write(str(datetime.now()) + "\n")

        print("Initializing Model...")
        colorizationModel = FusionModel(config)
        print("Model Initialized!")

        print("Start training")
        colorizationModel.train(train_data, test_data, logger, config=config)
