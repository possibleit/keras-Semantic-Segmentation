
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import show_label


class DataLoader(object):
    def __init__(self, train_path, val_path, img_folder, label_folder, val_img_folder, val_label_folder,
                 num_classes, image_size, label_size, batch_size=1, color_mode='rgb',
                 seed=19, image_save_prefix="image", label_save_prefix="label", save_to_dir=None, ):
        """[keras中导入数据]

        Arguments:
            train_path {[str]} -- [训练集路径]
            val_path {[str]} -- [验证集路径]
            img_folder {[str]} -- [训练集图片文件夹]
            label_folder {[str]} -- [训练集标签文件夹]
            val_img_folder {[str]} -- [验证集图片文件夹]
            val_label_folder {[str]} -- [验证集标签文件夹]
            num_classes {[int]} -- [类别数目]
            image_size {[tuple]} -- [训练图片resize的尺寸，如（512,512）]
            label_size {[tuple]} -- [标签resize的尺寸，如（512,512）]

        Keyword Arguments:
            batch_size {int} -- [批量大小] (default: {1})
            color_mode {str} -- [颜色模式] (default: {'rgb'})
        """
        self.train_path = train_path
        self.val_path = val_path
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.val_img_folder = val_img_folder
        self.val_label_folder = val_label_folder
        self.num_classes = num_classes
        # self.target_size = target_size
        self.image_size = image_size
        self.label_size = label_size
        self.color_mode = color_mode
        self.seed = seed
        self.batch_size = batch_size
        self.image_save_prefix = image_save_prefix
        self.label_save_prefix = label_save_prefix
        self.save_to_dir = save_to_dir
        self.data_gen_args = dict(
            # 数据提升时图片随机转动的角度
            rotation_range=0.2,
            # 数据提升时图片水平偏移的幅度
            width_shift_range=0.05,
            # 数据提升时图片竖直偏移的幅度
            height_shift_range=0.05,
            # 剪切强度（逆时针方向的剪切变换角度）
            shear_range=0.05,
            # 随机缩放的幅度
            zoom_range=0.05,
            # 进行随机竖直翻转
            vertical_flip=True,
            # 进行随机水平翻转
            horizontal_flip=True,
            # 当进行变换时超出边界的点将根据本参数给定的方法进行处理
            fill_mode='nearest'
        )

    def trainData(self):

        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)

        image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes=[self.img_folder],
            class_mode=None,
            color_mode=self.color_mode,
            target_size=self.image_size,
            batch_size=self.batch_size,
            save_to_dir=self.save_to_dir,
            save_prefix=self.image_save_prefix,
            seed=self.seed
        )

        label_generator = label_datagen.flow_from_directory(
            self.train_path,
            classes=[self.label_folder],
            class_mode=None,
            color_mode=self.color_mode,
            target_size=self.label_size,
            batch_size=self.batch_size,
            save_to_dir=self.save_to_dir,
            save_prefix=self.image_save_prefix,
            seed=self.seed
        )
        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:
            img, label = self.adjust_data(img, label)
            yield (img, label)

    def validData(self):
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.val_path,
            classes=[self.val_img_folder],
            class_mode=None,
            color_mode=self.color_mode,
            target_size=self.image_size,
            batch_size=self.batch_size,
            seed=self.seed
        )
        label_generator = label_datagen.flow_from_directory(
            self.val_path,
            classes=[self.val_label_folder],
            class_mode=None,
            color_mode=self.color_mode,
            target_size=self.label_size,
            batch_size=self.batch_size,
            seed=self.seed
        )
        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:
            img, label = self.adjust_data(img, label)
            yield (img, label)

    def adjust_data(self, img, label):
        img = img / 255.
        label = label[:, :, :, 0] if (
            len(label.shape) == 4) else label[:, :, 0]
        new_label = np.zeros(label.shape + (self.num_classes,))
        for i in range(self.num_classes):
            new_label[label == i, i] = 1
        label = new_label
        return (img, label)


# train_path = 'G:\备份\备份\dataset\CamVid'
# val_path = 'G:\备份\备份\dataset\CamVid'
# train_img_folder = 'train'
# train_label_folder = 'trainannot'
# val_image_folder = 'val'
# val_label_folder = 'valannot'
# image_size_ = (480, 480)
# label_size_ = (240, 240)
# image_size = (224, 224)
# label_size = (224, 224)
# color_mode = 'rgb'
# num_classes = 12

# dataloader1 = DataLoader(train_path, val_path, train_img_folder,
#                          train_label_folder, val_image_folder, val_label_folder,
#                          num_classes, image_size, label_size, batch_size=5)
# dataloader2 = DataLoader(train_path, val_path, train_img_folder,
#                          train_label_folder, val_image_folder, val_label_folder,
#                          num_classes, image_size_, label_size_, batch_size=5)

# data1 = dataloader1.trainData()
# data2 = dataloader2.trainData()
# x, y = next(data1)
# x_, y_ = next(data2)

# plt.subplot(1, 4, 1)
# plt.imshow(x[0])
# plt.subplot(1, 4, 2)
# show_label(y[0], label_size)

# plt.subplot(1, 4, 3)
# plt.imshow(np.squeeze(x[1]))
# plt.subplot(1, 4, 4)
# show_label(y[1], label_size)
# plt.show()
