#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
因为keras的api不支持索引传输，因此最大池化索引上采样被UpSampling2D替代
'''

import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization,  MaxPool2D, Dense, Softmax, UpSampling2D,Reshape
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras import Input

def encoder(input_shape):
    inputs = Input(input_shape)
    o = Conv2D(filters=64, kernel_size=3,activation='relu', padding='SAME',kernel_initializer='he_normal', name='block1_conv1')(inputs)
    o = Conv2D(filters=64, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block1_conv2')(o)
    o = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(o)
    f1 = o
    o = Conv2D(filters=128, kernel_size=3,activation='relu', padding='SAME',kernel_initializer='he_normal', name='block2_conv1')(o)
    o = Conv2D(filters=128, kernel_size=3,activation='relu', padding='SAME',kernel_initializer='he_normal', name='block2_conv2')(o)
    o = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(o)
    f2 = o
    o = Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block3_conv1')(o)
    o = Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block3_conv2')(o)
    o = Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block3_conv3')(o)
    o = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(o)
    f3 = o
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block4_conv1')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block4_conv2')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block4_conv3')(o)
    o = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(o)
    f4 = o
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block5_conv1')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block5_conv2')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal', name='block5_conv3')(o)
    o = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(o)
    f5 = o

    return inputs, [f1, f2, f3, f4, f5]


def decoder(num_class, features):
    o = features[-1]

    o = UpSampling2D((2,2))(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = Conv2D(filters=128, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(filters=128, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = Conv2D(filters=64, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(filters=64, kernel_size=3, activation='relu', padding='SAME',kernel_initializer='he_normal')(o)
    o = Conv2D(filters=num_class, kernel_size=1, activation='softmax', padding='SAME', kernel_initializer='he_normal')(o)

    return o


def _segnet(input_height, input_width, channel, num_class):
    inputs, features = encoder((input_height,input_width,channel))
    outputs = decoder(num_class, features)
#     outputs = Reshape((input_height*input_width, -1))(outputs)
#     outputs =Softmax()(o)

    model = Model(inputs,outputs)

    return model


def segnet(input_height=224, input_width=224, channel=3, num_class=2):
    model = _segnet(input_height,input_width, channel, num_class)
    return model


def print_model():
    model = _segnet(224, 224, 3, 2)
    model.summary()
    plot_model(model, to_file='SegNet.png', show_shapes=True)

