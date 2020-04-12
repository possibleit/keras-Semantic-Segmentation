#!/usr/bin/env python
# coding: utf-8

import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import cv2
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

BackGround = [255, 255, 255]
road = [0, 0, 0]
# COLOR_DICT = np.array([BackGround, road])
one = [128, 128, 128]
two = [128, 0, 0]
three = [192, 192, 128]
four = [255, 69, 0]
five = [128, 64, 128]
six = [60, 40, 222]
seven = [128, 128, 0]
eight = [192, 128, 128]
nine = [64, 64, 128]
ten = [64, 0, 128]
eleven = [64, 64, 0]
twelve = [0, 128, 192]
COLOR_DICT = np.array([one, two,three,four,five,six,seven,eight,nine,ten,eleven,twelve])


train_path = '/home/possibleit/文档/dataset/CamVid'
# train_path = 'K:/备份/dataset/CamVid'
image_folder = 'train'
label_folder = 'trainannot'
valid_path = '/home/possibleit/文档/dataset/CamVid'
# valid_path = 'K:/备份/dataset/CamVid'
valid_image_folder = 'val'
valid_label_folder = 'valannot'
log_dir = './log'
test_path = '/home/possibleit/文档/dataset/CamVid/test'
flag_multi_class = True
num_class = 12
image_color_mode = 'rgb'
label_color_mode = 'rgb'
target_size = (512, 512)
img_type = 'png'

data_gen_args = dict(rotation_range=0.2,
                                  width_shift_range=0.05,
                                  height_shift_range=0.05,
                                  shear_range=0.05,
                                  zoom_range=0.05,
                                  vertical_flip=True,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

def adjustData(img, label, flag_multi_class=True):
    if (flag_multi_class):
        img = img / 255.
        label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
        new_label = np.zeros(label.shape + (num_class,))
        for i in range(num_class):
            new_label[label == i, i] = 1
        label = new_label
    elif (np.max(img) > 1):
        img = img / 255.
        label = label / 255.
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    return (img, label)





def trainGenerator(batch_size, image_save_prefix="image", label_save_prefix="label",
                   save_to_dir=None, seed=7):
    image_datagen = ImageDataGenerator(**data_gen_args)
    label_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(
            train_path,
            classes=[image_folder],
            class_mode=None,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed)
    label_generator = label_datagen.flow_from_directory(
            train_path,
            classes=[label_folder],
            class_mode=None,
            color_mode=label_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=label_save_prefix,
            seed=seed)
    train_generator = zip(image_generator, label_generator)
    for (img, label) in train_generator:
        img, label = adjustData(img, label)
        yield (img, label)




def testGenerator():
    filenames = os.listdir(test_path)
    for filename in filenames:
        img = io.imread(os.path.join(test_path, filename), as_gray=False)
        img = img / 255.
        img = trans.resize(img, target_size, mode='constant')
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def validLoad(batch_size, seed=7):
    image_datagen = ImageDataGenerator(data_gen_args)
    label_datagen = ImageDataGenerator(data_gen_args)
    image_generator = image_datagen.flow_from_directory(
            valid_path,
            classes=[valid_image_folder],
            class_mode=None,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            seed=seed)
    label_generator = label_datagen.flow_from_directory(
            valid_path,
            classes=[valid_label_folder],
            class_mode=None,
            color_mode=label_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            seed=seed)
    train_generator = zip(image_generator, label_generator)
    for (img, label) in train_generator:
        img, label = adjustData(img, label)
        yield (img, label)

