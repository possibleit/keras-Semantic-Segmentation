from utils import COLOR_DICT, show_label, show_predict_image
from dataloader import DataLoader
from model.Unet import Unet
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import cv2
print(tf.__version__)
# from model.DeconvNet import DeconvNet
# from model.SegNet import SegNet
# from model.PSPNet import PSPNet
# from model.Deeplab import Deeplab

model = Unet((512, 512, 3), 12)
# model = DeconvNet((512, 512, 3), 12)
# model = PSPNet((480, 480, 3), 12)
# model = Deeplab((512, 512, 3), 12)
weight_path = 'Log//' + model.name + '//weight.h5'


model.load_weights(weight_path)

train_path = 'G:\备份\备份\dataset\CamVid'
val_path = 'G:\备份\备份\dataset\CamVid'
train_img_folder = 'train'
train_label_folder = 'trainannot'
val_image_folder = 'val'
val_label_folder = 'valannot'
image_size = (512, 512)
label_size = (512, 512)
color_mode = 'rgb'
num_classes = 12

dataloader = DataLoader(train_path, val_path, train_img_folder,
                        train_label_folder, val_image_folder, val_label_folder,
                        num_classes, image_size, label_size, batch_size=5)

x, y = dataloader.validData().__next__()
for i in range(5):
    plt.imshow(x[i])

    plt.axis('off')
    plt.gcf().set_size_inches(512 / 100, 512 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.93,
                        left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    savefig('pic\\' + ' pic ' + str(i) + '.png ')

    show_label(np.expand_dims(y[i], 0), label_size)

    plt.axis('off')
    plt.gcf().set_size_inches(512 / 100, 512 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.93,
                        left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    savefig('pic\\' + 'label' + str(i) + '.png')

    y_ = model.predict(np.expand_dims(x[i], 0))
    show_predict_image(y_, label_size)

    plt.axis('off')
    plt.gcf().set_size_inches(512 / 100, 512 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.93,
                        left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    savefig('pic\\' + 'pre' + str(i) + '.png')
