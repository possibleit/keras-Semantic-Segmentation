import keras
from keras.layers import *
from keras import Model, Input
from keras.initializers import glorot_uniform
from keras.utils import plot_model
import tensorflow as tf


def my_upsampling(x, resize_factor=8, method=0):
    # 0：双线性插值。1：最近邻居法。2：双三次插值法。3：面积插值法
    # NHWC
    img_h = x.shape[1] * resize_factor
    img_w = x.shape[2] * resize_factor
    return tf.image.resize_images(x, (img_h, img_w), 0)


def identity_block(x, filters, dilation, BN=False):
    filters_1, filters_2, filters_3 = filters
    x_shortcut = x
    x = Conv2D(filters=filters_1, kernel_size=1, strides=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters_2, kernel_size=1, strides=(1, 1), padding='same',
               dilation_rate=dilation, kernel_initializer=glorot_uniform(seed=0))(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    # stage 3
    x = Conv2D(filters=filters_3, kernel_size=1, strides=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    # stage 4
    x = Add()([x, x_shortcut])
    x = Activation(activation='relu')(x)

    return x


def conv_block(x, filters, strides, dilation, BN=False):
    filters_1, filters_2, filters_3 = filters
    x_shortcut = x
    # stage 1
    x = Conv2D(filters=filters_1, kernel_size=1, strides=1, padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filters=filters_2, kernel_size=3, strides=strides, padding='same',
               dilation_rate=dilation, kernel_initializer=glorot_uniform(seed=0))(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters_3, kernel_size=1, strides=1, activation='relu', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    # stage 4
    x_shortcut = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=strides, padding='same',
                        kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
    # stage 5
    x = Add()([x, x_shortcut])
    x = Activation(activation='relu')(x)
    return x


def resnet50(input_shape, BN=False):
    inputs = Input(input_shape)
    x = Conv2D(filters=64, kernel_size=7, strides=2,
               padding='SAME', use_bias=False)(inputs)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    x = conv_block(x, filters=[64, 64, 256], strides=1, dilation=1)
    x = identity_block(x, filters=[64, 64, 256], dilation=1)
    x = identity_block(x, filters=[64, 64, 256], dilation=1)

    x = conv_block(x, filters=[128, 128, 512], strides=2, dilation=1)
    x = identity_block(x, filters=[128, 128, 512], dilation=1)
    x = identity_block(x, filters=[128, 128, 512],  dilation=1)
    x = identity_block(x, filters=[128, 128, 512], dilation=1)

    x = conv_block(x,  filters=[256, 256, 1024], strides=1, dilation=2)
    x = identity_block(x, filters=[256, 256, 1024], dilation=2)
    x = identity_block(x, filters=[256, 256, 1024], dilation=2)
    x = identity_block(x, filters=[256, 256, 1024], dilation=2)
    x = identity_block(x, filters=[256, 256, 1024], dilation=2)
    x = identity_block(x, filters=[256, 256, 1024], dilation=2)

    x = conv_block(x, filters=[512, 512, 2048], strides=1, dilation=4)
    x = identity_block(x, filters=[256, 256, 2048], dilation=4)
    x = identity_block(x, filters=[256, 256, 2048], dilation=4)

    return inputs, x


def build_pspnet(input_shape, num_classes, BN=False):
    inputs, x = resnet50(input_shape, BN)

    feature_map_size = input_shape[0] / 8
    pool_size = [
        feature_map_size / 1,
        feature_map_size / 2,
        feature_map_size / 3,
        feature_map_size / 6
    ]

    x_c1 = AveragePooling2D(
        pool_size=(pool_size[0], pool_size[0]), strides=(pool_size[0], pool_size[0]), name='ave_c1')(x)
    x_c1 = Conv2D(filters=512, kernel_size=1, strides=1,
                  padding='SAME', name='conv_c1')(x_c1)
    if BN:
        x_c1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x_c1)
    x_c1 = Activation(activation='relu')(x_c1)
    # x_c1=UpSampling2D(size=(pool_size[0], pool_size[0]), name='up_c1')(x_c1)
    x_c1 = Lambda(my_upsampling, arguments={
                  'resize_factor': pool_size[0]})(x_c1)

    x_c2 = AveragePooling2D(
        pool_size=(pool_size[1], pool_size[1]), strides=(pool_size[1], pool_size[1]), name='ave_c2')(x)
    x_c2 = Conv2D(filters=512, kernel_size=1, strides=1,
                  padding='SAME', name='conv_c2')(x_c2)
    if BN:
        x_c2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x_c2)
    x_c2 = Activation(activation='relu')(x_c2)
    # x_c2=UpSampling2D(size=(pool_size[1], pool_size[1]), name='up_c2')(x_c2)
    x_c2 = Lambda(my_upsampling, arguments={
                  'resize_factor': pool_size[1]})(x_c2)

    x_c3 = AveragePooling2D(
        pool_size=(pool_size[2], pool_size[2]), strides=(pool_size[2], pool_size[2]), name='ave_c3')(x)
    x_c3 = Conv2D(filters=512, kernel_size=1, strides=1,
                  padding='SAME', name='conv_c3')(x_c3)
    if BN:
        x_c3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x_c3)
    x_c3 = Activation(activation='relu')(x_c3)
    # x_c3 = UpSampling2D(size=(pool_size[2], pool_size[2]), name='up_c3')(x_c3)
    x_c3 = Lambda(my_upsampling, arguments={
                  'resize_factor': pool_size[2]})(x_c3)

    x_c4 = AveragePooling2D(
        pool_size=(pool_size[3], pool_size[3]), strides=(pool_size[3], pool_size[3]), name='ave_c4')(x)
    x_c4 = Conv2D(filters=512, kernel_size=1, strides=1,
                  padding='SAME', name='conv_c4')(x_c4)
    if BN:
        x_c4 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x_c4)
    x_c4 = Activation(activation='relu')(x_c4)
    # x_c4 = UpSampling2D(size=(pool_size[3], pool_size[3]), name='up_c4')(x_c4)
    x_c4 = Lambda(my_upsampling, arguments={
                  'resize_factor': pool_size[3]})(x_c4)

    x_c5 = Conv2D(filters=512, kernel_size=1, strides=1,
                  name='conv_c5', padding='SAME')(x)
    if BN:
        x_c5 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x_c5)
    x_c5 = Activation(activation='relu')(x_c5)

    x = Concatenate(axis=-1, name='concat')([x_c1, x_c2, x_c3, x_c4, x_c5])
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME',
               name='sum_conv_1_11')(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    # x=UpSampling2D(size=(4, 4))(x)
    x = Lambda(my_upsampling, arguments={'resize_factor': 4})(x)

    x = Conv2D(filters=num_classes, kernel_size=1, strides=1, padding='SAME', name='sum_conv_2',
               activation='softmax')(x)

    model = Model(inputs, x)
    return model


def PSPNet(input_shape, num_classes, BN=False):
    model = build_pspnet(input_shape, num_classes, BN)
    model.name = 'PSPNet'
    model.summary()
    return model


# model = PSPNet((480, 480, 3), 21)
# plot_model(model, to_file='PSPNet.png', show_shapes=True)
