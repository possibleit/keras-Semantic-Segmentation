from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(rotation_range=40,  # 随机旋转的度数
                             width_shift_range=0.2,  # 随机水平平移
                             height_shift_range=0.2,  # 随机垂直平移
                             rescale=1 / 255,  # 数据归一化
                             shear_range=20,  # 随机错切变换
                             zoom_range=0.2,  # 随机放大
                             horizontal_flip=True,  # 水平翻转
                             fill_mode='nearest',  # 填充方式
                             )

# 载入图片
image = load_img(r'G:\备份\备份\dataset\CamVid\tt\0006R0_f01800.png')
x = img_to_array(image)
print(x.shape)

x = np.expand_dims(x, 0)
print(x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='tmp', save_prefix='new_', save_format='jpeg'):
    i += 1
    if i == 5:
        break
