'''
@Descripttion: 
@version: 
@Author: sueRimn
@Date: 1970-01-01 08:00:00
@LastEditors: sueRimn
@LastEditTime: 2020-04-29 21:31:21
'''
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils import plot_model
import os

MODEL_NAME = 'Unet'


def my_upsampling(x, resize_factor=8, method=0):
    # 0：双线性插值。1：最近邻居法。2：双三次插值法。3：面积插值法
    # NHWC
    img_h = x.shape[1] * resize_factor
    img_w = x.shape[2] * resize_factor
    return tf.image.resize_images(x, (img_h, img_w), 0)


def build_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Lambda(my_upsampling, arguments={
        'resize_factor': 2})(drop5)
    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Lambda(my_upsampling, arguments={
        'resize_factor': 2})(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Lambda(my_upsampling, arguments={
        'resize_factor': 2})(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)
    up9 = Lambda(my_upsampling, arguments={
        'resize_factor': 2})(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)

    return inputs, conv9


def Unet(input_shape, num_class):
    inputs, output = build_unet(input_shape)
    output = Conv2D(filters=num_class, kernel_size=1,
                    activation='softmax', padding='SAME')(output)
    model = Model(inputs, output)
    model.name = 'Unet'
    model.summary()
    return model


model = Unet((512, 512, 3), 21)

# plot_model(model, to_file='UNet.png', show_shapes=True)
