import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Model, applications, Sequential
from keras.layers import Input, BatchNormalization, Dense, Flatten, RepeatVector, Reshape, UpSampling2D, \
    Conv2DTranspose, Activation, MaxPooling2D
from keras.layers import concatenate, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import _Merge, Concatenate, Add


def bg_colorization_network(shape):
    input_img = Input(shape=(*shape, 3))

    # VGG16 without top layers
    VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model_ = Model(VGG_model.input, VGG_model.layers[-6].output)
    vgg_feature = model_(tf.tile(input_img, [1, 1, 1, 3]))

    # Global features
    global_features = Sequential([
        Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),

        Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),
    ])(vgg_feature)

    # Global feature pass back to colorization + classification
    global_merge_back = Sequential([
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        RepeatVector(28 * 28),
        Reshape((28, 28, 256)),
    ])(global_features)

    global_feature_class = Sequential([
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax'),
    ])(global_features)

    # Mid-level features
    midlevel_features = Sequential([
        Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),
    ])(vgg_feature)

    # Fusion of (VGG16 -> Mid-level) + (VGG16 -> Global)
    fusion = concatenate([midlevel_features, global_merge_back])

    # Fusion + Colorization
    output = Sequential([
        Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu'),
        Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        UpSampling2D(size=(2, 2)),
        Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        UpSampling2D(size=(2, 2)),
        Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid'),
        UpSampling2D(size=(2, 2)),
    ])(fusion)

    generated = Model(input=input_img, outputs=[output, global_feature_class])

    return generated


def simplified_colorization_network(shape):
    input_img = Input(shape=shape)
    input_img_1 = Reshape((*shape, 1))(input_img)
    input_img_3 = Concatenate(axis=3)([input_img_1, input_img_1, input_img_1])

    # VGG16 without top layers
    VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    thumbnail1 = Model(VGG_model.input, VGG_model.layers[2].output)(input_img_3)
    thumbnail2 = Model(VGG_model.input, VGG_model.layers[5].output)(input_img_3)
    thumbnail4 = Model(VGG_model.input, VGG_model.layers[9].output)(input_img_3)
    thumbnail8 = Model(VGG_model.input, VGG_model.layers[13].output)(input_img_3)

    # Global features
    global_features = Sequential([
        Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu'),  # H/16
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu'),  # H/32
        BatchNormalization(),
    ])(thumbnail8)

    # Global feature pass back to colorization
    global_merge_back = Sequential([
        Flatten(),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        RepeatVector(28 * 28),
        Reshape((28, 28, 128)),
    ])(global_features)

    # Mid-level features
    mid_level_features = Sequential([
        Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),
        Conv2D(384, (3, 3), padding='same', strides=(1, 1), activation='relu'),
        BatchNormalization(),
    ])(thumbnail8)

    # Fusion of (VGG16 -> Mid-level) + (VGG16 -> Global) + Shortcut
    forge8 = concatenate([mid_level_features, global_merge_back])
    forge8_post = Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(forge8)
    thumbnail8_shortcut = Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(thumbnail8)
    forge8_sum_shortcut = Add()([forge8_post, thumbnail8_shortcut])

    forge4 = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2), activation='relu')(forge8_sum_shortcut)
    forge4_post = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(forge4)
    thumbnail4_shortcut = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(thumbnail4)
    forge4_sum_shortcut = Add()([forge4_post, thumbnail4_shortcut])

    forge2 = Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2), activation='relu')(forge4_sum_shortcut)
    forge2_post = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(forge2)
    thumbnail2_shortcut = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(thumbnail2)
    forge2_sum_shortcut = Add()([forge2_post, thumbnail2_shortcut])

    forge1 = Conv2DTranspose(16, (3, 3), padding='same', strides=(2, 2), activation='relu')(forge2_sum_shortcut)
    forge1_post = Conv2D(16, (3, 3), padding='same', strides=(1, 1), activation='relu')(forge1)
    thumbnail1_shortcut = Conv2D(16, (3, 3), padding='same', strides=(1, 1), activation='relu')(thumbnail1)
    forge1_sum_shortcut = Add()([forge1_post, thumbnail1_shortcut])
    output = Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')(forge1_sum_shortcut)

    generated = Model(input=input_img, output=output)

    return generated


def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    shortcut = Conv2D(nb_filter2, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def simple_instance_network(shape):
    input_img = Input(shape=shape)
    input_img_1 = Reshape((*shape, 1))(input_img)

    # Stage 1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv1', use_bias=True)(input_img_1)
    x = BatchNormalization(name='bn_conv1')(x)
    thumbnail1 = x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64], stage=2, block='a')
    thumbnail2 = x = identity_block(x, 3, [64, 64], stage=2, block='b')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 3
    x = conv_block(x, 3, [128, 128], stage=3, block='a')
    thumbnail4 = x = identity_block(x, 3, [128, 128], stage=3, block='b')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 4
    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    thumbnail8 = identity_block(x, 3, [256, 256], stage=4, block='b')

    # Co-stage 4
    forge8 = thumbnail8
    forge8 = conv_block(forge8, 3, [256, 256], stage=5, block='a')
    forge8 = identity_block(forge8, 3, [256, 256], stage=5, block='b')

    # Co-stage 3
    forge4 = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2), activation='relu')(forge8)
    forge4_post = identity_block(forge4, 3, [128, 128], stage=6, block='a')
    thumbnail4_shortcut = conv_block(thumbnail4, 3, [128, 128], stage=6, block='b')
    forge4_sum_shortcut = Add()([forge4_post, thumbnail4_shortcut])

    # Co-stage 2
    forge2 = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2), activation='relu')(forge4_sum_shortcut)
    forge2_post = identity_block(forge2, 3, [64, 64], stage=7, block='a')
    thumbnail2_shortcut = conv_block(thumbnail2, 3, [64, 64], stage=7, block='b')
    forge2_sum_shortcut = Add()([forge2_post, thumbnail2_shortcut])

    # Co-stage 1
    forge1 = Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2), activation='relu')(forge2_sum_shortcut)
    forge1_post = identity_block(forge1, 3, [32, 32], stage=8, block='a')
    thumbnail1_shortcut = conv_block(thumbnail1, 3, [32, 32], stage=8, block='b')
    forge1_sum_shortcut = Add()([forge1_post, thumbnail1_shortcut])

    output = Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')(forge1_sum_shortcut)

    generated = Model(input=input_img, output=output)

    return generated


def discriminator_network(shape):
    input_ab = Input(shape=(*shape, 2), name='d_ab_input')
    input_l = Input(shape=(*shape, 1), name='d_l_input')
    net = concatenate([input_l, input_ab], name='d_concat')
    net = Conv2D(64, (4, 4), padding='same', strides=(2, 2), name='d_conv2d_1')(net)  # 112, 112, 64
    net = LeakyReLU(name='leakyrelu_1')(net)
    net = Conv2D(128, (4, 4), padding='same', strides=(2, 2), name='d_conv2d_2')(net)  # 56, 56, 128
    net = LeakyReLU(name='leakyrelu_2')(net)
    net = Conv2D(256, (4, 4), padding='same', strides=(2, 2), name='d_conv2d_3')(net)  # 28, 28, 256
    net = LeakyReLU(name='leakyrelu_3')(net)
    net = Conv2D(512, (4, 4), padding='same', strides=(1, 1), name='d_conv2d_4')(net)  # 28, 28, 512
    net = LeakyReLU(name='leakyrelu_4')(net)
    net = Conv2D(1, (4, 4), padding='same', strides=(1, 1), name='d_conv2d_5')(net)  # 28, 28,1
    return Model([input_ab, input_l], net)



def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


def wasserstein_loss_dummy(y_true, y_pred):
    return tf.reduce_mean(y_pred)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((tf.shape(inputs[0])[0], 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
