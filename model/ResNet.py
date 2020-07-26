'''
@Descripttion: 
@version: 
@Author: sueRimn
@Date: 1970-01-01 08:00:00
@LastEditors: sueRimn
@LastEditTime: 2020-04-22 08:48:33
'''
# 暂定，需输出模型图片对比

from keras.layers import *
from keras import Model, Input
from keras.utils import plot_model


def identity_block(x, filters):
    """
    * 对应于50层101层152层的残差网络的残差结构中的普通残差结构
    * filters对应三个卷积层的卷积核的个数，如果捷径分支上有卷积，则捷径分支上的卷积核个数等于第三个卷积层的参数
    """

    filters_1, filters_2, filters_3 = filters

    x_shortcut = x
    x = Conv2D(filters=filters_1, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filters=filters_2, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filters=filters_3, kernel_size=1, strides=1,  padding='same')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    x = Add()([x, x_shortcut])
    x = Activation(activation='relu')(x)

    return x


def conv_block(x, filters, strides):
    """
    * 对应于50层101层152层的残差网络的残差结构中的虚线残差结构
    * filters对应三个卷积层的卷积核的个数，如果捷径分支上有卷积，则捷径分支上的卷积核个数等于第三个卷积层的参数
    * strides对应于下采样步骤，在conv3,conv4,conv5中第一个残差结构负责下采样操作，通过将卷积步幅设置为2来下采样的，
    * 在卷积核大小为3*3的那一层，也即第二层
    """
    filters_1, filters_2, filters_3 = filters
    x_shortcut = x
    x = Conv2D(filters=filters_1, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filters=filters_2, kernel_size=3,
               strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filters=filters_3, kernel_size=1, strides=1,
               activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    # 捷径分支
    x_shortcut = Conv2D(filters=filters_3, kernel_size=1,
                        strides=strides, padding='same')(x_shortcut)

    x = Add()([x, x_shortcut])
    x = Activation(activation='relu')(x)
    return x


def build_resnet50(input_shape):

    inputs = Input(input_shape)
    x = Conv2D(filters=64, kernel_size=7, strides=2, padding='SAME',
               activation='relu', use_bias=False)(inputs)
    x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    x = conv_block(x,  filters=[64, 64, 256], strides=1)
    x = identity_block(x,  filters=[64, 64, 256])
    x = identity_block(x,  filters=[64, 64, 256])

    x = conv_block(x,   filters=[128, 128, 512], strides=2)
    x = identity_block(x,  filters=[128, 128, 512])
    x = identity_block(x,   filters=[128, 128, 512])
    x = identity_block(x,   filters=[128, 128, 512])

    x = conv_block(x,   filters=[256, 256, 1024], strides=2)
    x = identity_block(x,   filters=[256, 256, 1024])
    x = identity_block(x,   filters=[256, 256, 1024])
    x = identity_block(x,   filters=[256, 256, 1024])
    x = identity_block(x,   filters=[256, 256, 1024])
    x = identity_block(x,   filters=[256, 256, 1024])

    x = conv_block(x,  filters=[512, 512, 2048], strides=2)
    x = identity_block(x,   filters=[256, 256, 2048])
    x = identity_block(x,   filters=[256, 256, 2048])

    model = Model(inputs, x)
    return model


def ResNet50(input_shape):
    model = build_resnet50(input_shape)
    model.name = 'resnet50'

    model.summary()

    return model


# model = ResNet50((480, 480, 3))
# plot_model(model, to_file='Resnet.png', show_shapes=True)
