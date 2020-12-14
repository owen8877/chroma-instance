import os
from datetime import datetime
from functools import partial
from math import ceil

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, applications, Sequential
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.engine.saving import load_model
from tensorflow.python.keras.layers import Input, BatchNormalization, Dense, Flatten, RepeatVector, Reshape, Lambda, \
    UpSampling2D, Layer, concatenate, Conv2D
from tensorflow.python.keras.losses import huber_loss
from tensorflow.python.keras.optimizers import Adam
from tqdm import tqdm

from chroma_instance.config import MAX_INSTANCES
from chroma_instance.config.FirstTest import FirstTestConfig
from chroma_instance.data.generator import Data
from chroma_instance.model.basic import discriminator_network, RandomWeightedAverage, wasserstein_loss_dummy, \
    gradient_penalty_loss, constant_loss_dummy
from chroma_instance.util import write_log, deprocess_float2int, reconstruct_and_save, prepare_logger

GRADIENT_PENALTY_WEIGHT = 10


class WeightGenerator(Layer):
    def __init__(self, units, batch_size, **kwargs):
        super(WeightGenerator, self).__init__(**kwargs)

        self.instance_conv = Sequential([
            Conv2D(units, 3, padding='same', strides=1, activation='relu', name='weight_instance_conv1'),
            Conv2D(units, 3, padding='same', strides=1, activation='relu', name='weight_instance_conv2'),
            Conv2D(1, 3, padding='same', strides=1, activation='relu', name='weight_instance_conv3'),
        ])

        self.simple_bg_conv = Sequential([
            Conv2D(units, 3, padding='same', strides=1, activation='relu', name='weight_bg_conv1'),
            Conv2D(units, 3, padding='same', strides=1, activation='relu', name='weight_bg_conv2'),
            Conv2D(1, 3, padding='same', strides=1, activation='relu', name='weight_bg_conv3'),
        ])

        self.units = units
        self.batch_size = batch_size

    def resize_and_pad(self, feauture_maps, info_array, bg_size):
        y_size = tf.math.maximum(tf.cast(bg_size[0] * (info_array[1] - info_array[0]), tf.int32), 1)
        x_size = tf.math.maximum(tf.cast(bg_size[1] * (info_array[3] - info_array[2]), tf.int32), 1)
        feauture_maps = tf.image.resize(feauture_maps, (y_size, x_size))
        y_front_pad = tf.cast(bg_size[0] * info_array[0], tf.int32)
        x_front_pad = tf.cast(bg_size[1] * info_array[2], tf.int32)
        padding = [
            [y_front_pad, bg_size[0] - y_front_pad - y_size],
            [x_front_pad, bg_size[1] - x_front_pad - x_size],
            [0, 0],
        ]
        feauture_maps = tf.pad(feauture_maps, padding)
        return feauture_maps

    def call(self, inputs, **kwargs):
        instance_feature, bg_feature, bbox, mrcnn_mask = inputs
        bg_size = bg_feature.get_shape().as_list()[1:3]

        output = []

        combined_instance_feature = pack_instance(instance_feature)
        combined_instance_weight = self.instance_conv(combined_instance_feature)
        instance_weights = unpack_instance(combined_instance_weight)

        bg_weights = self.simple_bg_conv(bg_feature)

        for i in range(self.batch_size):
            resized_sub_weights = []
            resized_masks = []
            for j in range(MAX_INSTANCES):
                box_info = bbox[i, :, j]  # (4, )
                instance_weight = instance_weights[i, :, :, :, j]  # (h, w, 1)
                resized_sub_weight = self.resize_and_pad(instance_weight, box_info, bg_size)  # (h_i, w_i, 1)

                instance_mask = mrcnn_mask[i, :, :, j:j + 1]  # (H, W, 1)
                resized_mask = self.resize_and_pad(instance_mask, [0, 1, 0, 1], bg_size)  # (h_i, w_i, 1)

                resized_sub_weights.append(resized_sub_weight)
                resized_masks.append(resized_mask)

            bg_weight = bg_weights[i, :, :, :]

            exp_weight = []
            for f, m in zip(resized_sub_weights, resized_masks):
                exp_weight.append(tf.exp(f) * m)
            exp_weight.append(tf.exp(bg_weight))

            exp_weight_sum = sum(exp_weight)
            softmax_weight = [w / exp_weight_sum for w in exp_weight]

            composed = softmax_weight[-1] * bg_feature[i, :, :, :]
            for j in range(MAX_INSTANCES):
                box_info = bbox[i, :, j]
                instance_sub_feature = instance_feature[i, :, :, :, j]  # (h, w, c)
                resized_sub_feature = self.resize_and_pad(instance_sub_feature, box_info, bg_size)  # (h_i, w_i, c)
                composed += resized_sub_feature * softmax_weight[j]

            output.append(composed)

        stacked_output = tf.stack(output)
        return stacked_output

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super(WeightGenerator, self).get_config()
        config.update({
            # "instance_weights": self.instance_conv.get_weights(),
            # "bg_weights": self.simple_bg_conv.get_weights(),
            "units": self.units,
            "batch_size": self.batch_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        obj = cls(config["units"], config["batch_size"])
        # obj.instance_conv.set_weights(config["instance_weights"])
        # obj.simple_bg_conv.set_weights(config["bg_weights"])
        return obj


def pack_instance(x):
    return tf.squeeze(
        tf.reshape(
            tf.transpose(
                x,
                perm=[0, 4, 1, 2, 3]),
            (-1, 1, *x.get_shape().as_list()[1:-1])),
        axis=1)


def unpack_instance(x):
    return tf.transpose(
        tf.reshape(
            tf.expand_dims(
                x,
                axis=1),
            (-1, MAX_INSTANCES, *x.get_shape().as_list()[1:])),
        perm=[0, 2, 3, 4, 1])


def instance_network(shape):
    input_img = Input(shape=(*shape, 1, MAX_INSTANCES), name='fg_input_img')

    input_packed = Lambda(lambda x: pack_instance(x), name='fg_pack_input')(input_img)
    input_img_3 = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]), name='fg_input_tile')(input_packed)

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
    up_sampling2d_1 = UpSampling2D(size=(2, 2), name='fg_up_sampling2d_1', interpolation='bilinear')(conv2d_13)
    # conv2dt_1 = Conv2DTranspose(64, (4, 4), padding='same', strides=(2, 2), name='fg_conv2dt_1')(conv2d_13)

    conv2d_14 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_14')(
        up_sampling2d_1)
    conv2d_15 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_15')(conv2d_14)
    up_sampling2d_2 = UpSampling2D(size=(2, 2), name='fg_up_sampling2d_2', interpolation='bilinear')(conv2d_15)
    # conv2dt_2 = Conv2DTranspose(32, (4, 4), padding='same', strides=(2, 2), name='fg_conv2dt_2')(conv2d_15)

    conv2d_16 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu', name='fg_conv2d_16')(
        up_sampling2d_2)
    conv2d_17 = Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid', name='fg_conv2d_17')(conv2d_16)
    up_sampling2d_3 = UpSampling2D(size=(2, 2), name='fg_up_sampling2d_3')(conv2d_17)

    model_3_unpack = Lambda(lambda x: unpack_instance(x), name='fg_model_3_unpack')(model_3)
    conv2d_11_unpack = Lambda(lambda x: unpack_instance(x), name='fg_conv2d_11_unpack')(conv2d_11)
    conv2d_13_unpack = Lambda(lambda x: unpack_instance(x), name='fg_conv2d_13_unpack')(conv2d_13)
    conv2d_15_unpack = Lambda(lambda x: unpack_instance(x), name='fg_conv2d_15_unpack')(conv2d_15)
    conv2d_17_unpack = Lambda(lambda x: unpack_instance(x), name='fg_conv2d_17_unpack')(conv2d_17)
    up_sampling2d_3_unpack = Lambda(lambda x: unpack_instance(x), name='up_sampling2d_3_unpack')(up_sampling2d_3)
    generated = Model(inputs=input_img,
                      outputs=[model_3_unpack, conv2d_11_unpack, conv2d_13_unpack, conv2d_15_unpack, conv2d_17_unpack,
                               up_sampling2d_3_unpack])

    return generated


def fusion_network(shape, batch_size):
    input_img = Input(shape=(*shape, 1), name='input_img')
    input_img_3 = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]), name='input_tile')(input_img)

    bbox = Input(shape=(4, MAX_INSTANCES), name='bbox')
    mask = Input(shape=(*shape, MAX_INSTANCES), name='mask')

    # VGG16 without top layers
    VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_model_3_pre = Model(VGG_model.input, VGG_model.layers[-6].output, name='model_3')(input_img_3)
    fg_model_3 = Input(shape=(*vgg_model_3_pre.get_shape().as_list()[1:], MAX_INSTANCES), name='fg_model_3')  # <-
    vgg_model_3 = WeightGenerator(64, batch_size, name='weight_generator_1')(
        [fg_model_3, vgg_model_3_pre, bbox, mask])  # <-

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
    fg_conv2d_11 = Input(shape=(*conv2d_11_pre.get_shape().as_list()[1:], MAX_INSTANCES), name='fg_conv2d_11')  # <-
    conv2d_11 = WeightGenerator(32, batch_size, name='weight_generator_2')(
        [fg_conv2d_11, conv2d_11_pre, bbox, mask])  # <-
    batch_normalization_6 = BatchNormalization(name='batch_normalization_6')(conv2d_11)

    # Fusion of (VGG16 -> Mid-level) + (VGG16 -> Global) + Colorization
    concatenate_2 = concatenate([batch_normalization_6, reshape_1], name='concatenate_2')

    conv2d_12 = Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv2d_12')(concatenate_2)
    conv2d_13_pre = Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_13')(conv2d_12)
    fg_conv2d_13 = Input(shape=(*conv2d_13_pre.get_shape().as_list()[1:], MAX_INSTANCES), name='fg_conv2d_13')  # <-
    conv2d_13 = WeightGenerator(16, batch_size, name='weight_generator_3')(
        [fg_conv2d_13, conv2d_13_pre, bbox, mask])  # <-
    # conv2dt_1 = Conv2DTranspose(64, (4, 4), padding='same', strides=(2, 2), name='conv2dt_1')(conv2d_13)
    up_sampling2d_1 = UpSampling2D(size=(2, 2), name='up_sampling2d_1', interpolation='bilinear')(conv2d_13)

    conv2d_14 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_14')(up_sampling2d_1)
    conv2d_15_pre = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_15')(conv2d_14)
    fg_conv2d_15 = Input(shape=(*conv2d_15_pre.get_shape().as_list()[1:], MAX_INSTANCES), name='fg_conv2d_15')  # <-
    conv2d_15 = WeightGenerator(16, batch_size, name='weight_generator_4')(
        [fg_conv2d_15, conv2d_15_pre, bbox, mask])  # <-
    # conv2dt_2 = Conv2DTranspose(32, (4, 4), padding='same', strides=(2, 2), name='conv2dt_2')(conv2d_15)
    up_sampling2d_2 = UpSampling2D(size=(2, 2), name='up_sampling2d_2', interpolation='bilinear')(conv2d_15)

    conv2d_16 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv2d_16')(up_sampling2d_2)
    conv2d_17_pre = Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid', name='conv2d_17')(conv2d_16)
    fg_conv2d_17 = Input(shape=(*conv2d_17_pre.get_shape().as_list()[1:], MAX_INSTANCES), name='fg_conv2d_17')  # <-
    conv2d_17 = WeightGenerator(16, batch_size, name='weight_generator_5')(
        [fg_conv2d_17, conv2d_17_pre, bbox, mask])  # <-
    # conv2dt_3 = Conv2DTranspose(2, (4, 4), padding='same', strides=(2, 2), name='conv2dt_3')(conv2d_17)
    up_sampling2d_3 = UpSampling2D(size=(2, 2), name='up_sampling2d_3', interpolation='bilinear')(conv2d_17)

    return Model(
        inputs=[input_img, fg_model_3, fg_conv2d_11, fg_conv2d_13, fg_conv2d_15, fg_conv2d_17, bbox, mask],
        outputs=[up_sampling2d_3, dense_6])


class FusionModel:
    def __init__(self, config, load_weight_path=None, ab_loss='mse'):
        img_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE)

        # Creating generator and discriminator
        optimizer = Adam(0.00002, 0.5)

        self.foreground_generator = instance_network(img_shape)
        self.foreground_generator.compile(
            loss=[constant_loss_dummy, constant_loss_dummy, constant_loss_dummy, constant_loss_dummy,
                  constant_loss_dummy, ab_loss], optimizer=optimizer)

        self.fusion_discriminator = discriminator_network(img_shape)
        self.fusion_discriminator.compile(loss=wasserstein_loss_dummy, optimizer=optimizer)
        self.fusion_generator = fusion_network(img_shape, config.BATCH_SIZE)
        self.fusion_generator.compile(loss=[ab_loss, 'kld'], optimizer=optimizer)

        if load_weight_path:
            chroma_gan = load_model(load_weight_path)
            chroma_gan_layers = [layer.name for layer in chroma_gan.layers]

            print('Loading chroma GAN parameter to instance network...')
            instance_layer_names = [layer.name for layer in self.foreground_generator.layers]
            for i, layer in enumerate(instance_layer_names):
                if layer == 'fg_model_3':
                    print('model 3 skip')
                    continue
                if len(layer) < 2:
                    continue
                if layer[:3] == 'fg_':
                    try:
                        j = chroma_gan_layers.index(layer[3:])
                        self.foreground_generator.layers[i].set_weights(chroma_gan.layers[j].get_weights())
                        print(f'Successfully set weights for layer {layer}')
                    except ValueError:
                        print(f'Layer {layer} not found in chroma gan.')
                    except Exception as e:
                        print(e)

            print('Loading chroma GAN parameter to fusion network...')
            fusion_layer_names = [layer.name for layer in self.fusion_generator.layers]
            for i, layer in enumerate(fusion_layer_names):
                if layer == 'model_3':
                    print('model 3 skip')
                    continue
                try:
                    j = chroma_gan_layers.index(layer)
                    self.fusion_generator.layers[i].set_weights(chroma_gan.layers[j].get_weights())
                    print(f'Successfully set weights for layer {layer}')
                except ValueError:
                    print(f'Layer {layer} not found in chroma gan.')
                except Exception as e:
                    print(e)

        # Fg=instance prediction
        fg_img_l = Input(shape=(*img_shape, 1, MAX_INSTANCES), name='fg_img_l')

        # self.foreground_generator.trainable = False
        self.foreground_generator.trainable = False
        fg_model_3, fg_conv2d_11, fg_conv2d_13, fg_conv2d_15, fg_conv2d_17, up_sampling2d_3 = self.foreground_generator(
            fg_img_l)

        # Fusion prediction
        fusion_img_l = Input(shape=(*img_shape, 1), name='fusion_img_l')
        fusion_img_real_ab = Input(shape=(*img_shape, 2), name='fusion_img_real_ab')
        fg_bbox = Input(shape=(4, MAX_INSTANCES), name='fg_bbox')
        fg_mask = Input(shape=(*img_shape, MAX_INSTANCES), name='fg_mask')

        self.fusion_generator.trainable = False
        fusion_img_pred_ab, fusion_class_vec = self.fusion_generator([fusion_img_l, fg_model_3, fg_conv2d_11,
                                                                      fg_conv2d_13, fg_conv2d_15, fg_conv2d_17, fg_bbox,
                                                                      fg_mask])

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
            inputs=[fusion_img_l, fusion_img_real_ab, fg_img_l, fg_bbox, fg_mask],
            outputs=[dis_real_ab,
                     dis_pred_ab,
                     dis_interp_ab], name='discriminator')

        self.discriminator_model.compile(optimizer=optimizer,
                                         loss=[wasserstein_loss_dummy,
                                               wasserstein_loss_dummy,
                                               partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])

        self.fusion_generator.trainable = True
        self.fusion_discriminator.trainable = False
        self.combined = Model(inputs=[fusion_img_l, fg_img_l, fg_bbox, fg_mask],
                              outputs=[fusion_img_pred_ab, up_sampling2d_3, fusion_class_vec, dis_pred_ab],
                              name='combined')
        self.combined.compile(loss=[ab_loss, ab_loss, 'kld', wasserstein_loss_dummy],
                              loss_weights=[1.0, 0.5, 0.003, -0.1],
                              optimizer=optimizer)

        # Monitor stuff
        self.callback = TensorBoard(config.LOG_DIR)
        self.callback.set_model(self.combined)
        self.train_names = ['loss', 'mse_loss', 'kullback_loss', 'wasserstein_loss']
        self.disc_names = ['disc_loss', 'disc_valid', 'disc_fake', 'disc_gp']

        self.test_loss_array = []
        self.g_loss_array = []

    def train(self, data: Data, test_data, log, config, skip_to_after_epoch=None):
        # Load VGG network
        VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)

        # Real, Fake and Dummy for Discriminator
        positive_y = np.ones((data.batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((data.batch_size, 1), dtype=np.float32)

        # total number of batches in one epoch
        total_batch = int(data.size / data.batch_size)
        print(f'batch_size={data.batch_size} * total_batch={total_batch}')

        save_path = lambda type, epoch: os.path.join(config.MODEL_DIR, f"fusion_{type}Epoch{epoch}.h5")

        if skip_to_after_epoch:
            start_epoch = skip_to_after_epoch + 1
            print(f"Loading weights from epoch {skip_to_after_epoch}")
            self.combined.load_weights(save_path("combined", skip_to_after_epoch))
            self.fusion_discriminator.load_weights(save_path("discriminator", skip_to_after_epoch))
        else:
            start_epoch = 0

        for epoch in range(start_epoch, config.NUM_EPOCHS):
            for batch in tqdm(range(total_batch)):
                train_batch = data.generate_batch()
                resized_l = train_batch.resized_images.l
                resized_ab = train_batch.resized_images.ab

                # GT vgg
                predictVGG = VGG_modelF.predict(np.tile(resized_l, [1, 1, 1, 3]))

                # train generator
                g_loss = self.combined.train_on_batch(
                    [resized_l, train_batch.instances.l, train_batch.instances.bbox, train_batch.instances.mask],
                    [resized_ab, train_batch.instances.ab, predictVGG, positive_y])
                # train discriminator
                d_loss = self.discriminator_model.train_on_batch(
                    [resized_l, resized_ab, train_batch.instances.l, train_batch.instances.bbox,
                     train_batch.instances.mask],
                    [positive_y, negative_y, dummy_y])

                # update log files
                write_log(self.callback, self.train_names, g_loss, (epoch * total_batch + batch + 1))
                write_log(self.callback, self.disc_names, d_loss, (epoch * total_batch + batch + 1))

                if batch % 10 == 0:
                    print(
                        f"[Epoch {epoch}] [Batch {batch}/{total_batch}] [generator loss: {g_loss[0]:08f}] [discriminator loss: {d_loss[0]:08f}]")

            print('Saving models...')
            self.combined.save(save_path("combined", epoch))
            self.fusion_discriminator.save(save_path("discriminator", epoch))
            print('Models saved.')

            print('Sampling test images...')
            # sample images after each epoch
            self.sample_images(test_data, epoch, config)

    def sample_images(self, test_data: Data, epoch, config):
        total_batch = int(ceil(test_data.size / test_data.batch_size))
        for _ in range(total_batch):
            # load test data
            test_batch = test_data.generate_batch()

            # predict AB channels
            fg_model_3, fg_conv2d_11, fg_conv2d_13, fg_conv2d_15, fg_conv2d_17, up_sampling2d_3 = self.foreground_generator.predict(
                test_batch.instances.l)

            fusion_img_pred_ab, _ = self.fusion_generator.predict(
                [test_batch.resized_images.l, fg_model_3, fg_conv2d_11, fg_conv2d_13, fg_conv2d_15, fg_conv2d_17,
                 test_batch.instances.bbox, test_batch.instances.mask])

            # print results
            for i in range(test_data.batch_size):
                original_full_img = test_batch.images.full[i]
                height, width, _ = original_full_img.shape
                pred_ab = cv2.resize(deprocess_float2int(fusion_img_pred_ab[i]), (width, height))
                reconstruct_and_save(test_batch.images.l[i], pred_ab, f'epoch{epoch}_{test_batch.file_names[i]}',
                                     config)


if __name__ == '__main__':
    config_name = 'fusion_2obj_huber'
    if config_name == 'fusion_2obj':
        config = FirstTestConfig('fusion_2obj', ROOT_DIR='../../../')
        train_data = Data(config.TRAIN_DIR, config)
        test_data = Data(config.TEST_DIR, config)
        with prepare_logger(train_data.batch_size, config) as logger:
            logger.write(str(datetime.now()) + "\n")

            # Scenario 1: fresh start
            # print("Initializing Model...")
            # colorizationModel = FusionModel(config, load_weight_path='../../../weights/chroma_gan/imagenet.h5')
            # print("Model Initialized!")
            # print("Start training")
            # colorizationModel.train(train_data, test_data, logger, config=config)

            # Scenario 2: pick up where we left
            print("Initializing Model...")
            colorizationModel = FusionModel(config)
            print("Model Initialized!")
            print("Start training")
            colorizationModel.train(train_data, test_data, logger, config=config, skip_to_after_epoch=1)
    elif config_name == 'fusion_2obj_huber':
        config = FirstTestConfig('fusion_2obj_huber', ROOT_DIR='../../../')
        train_data = Data(config.TRAIN_DIR, config)
        test_data = Data(config.TEST_DIR, config)
        with prepare_logger(train_data.batch_size, config) as logger:
            logger.write(str(datetime.now()) + "\n")

            # Scenario 1: fresh start
            print("Initializing Model...")
            colorizationModel = FusionModel(config, load_weight_path='../../../weights/chroma_gan/imagenet.h5',
                                            ab_loss=lambda y_true, y_pred: tf.reduce_mean(huber_loss(y_true, y_pred, delta=0.5)))
            print("Model Initialized!")
            print("Start training")
            colorizationModel.train(train_data, test_data, logger, config=config)

            # Scenario 2: pick up where we left
            # print("Initializing Model...")
            # colorizationModel = FusionModel(config)
            # print("Model Initialized!")
            # print("Start training")
            # colorizationModel.train(train_data, test_data, logger, config=config, skip_to_after_epoch=1)
