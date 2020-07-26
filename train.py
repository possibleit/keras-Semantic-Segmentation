'''
@Descripttion: 
@version: 
@Author: sueRimn
@Date: 1970-01-01 08:00:00
@LastEditors: sueRimn
@LastEditTime: 2020-05-01 21:19:23
'''

from model.Deeplab import Deeplab

from iou import mean_iou
from dataloader import DataLoader
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

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
                        num_classes, image_size, label_size)

model = Deeplab((512, 512, 3), 12)
log_dir = 'Log\\' + model.name

tb_cb = TensorBoard(log_dir=log_dir,
                    write_graph=True,
                    write_grads=False,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None,
                    embeddings_data=None,
                    update_freq='epoch')
model_checkpoint = ModelCheckpoint(filepath=log_dir +
                                   "/weights_{epoch:02d}-{val_loss:.4f}.h5",
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True,
                                   period=3)
rl = ReduceLROnPlateau(monitor='val_loss',
                       factor=0.5,
                       patience=3,
                       verbose=1)
csv_logger = CSVLogger(log_dir + '\\training.log')

callbacks = [model_checkpoint, tb_cb, rl, csv_logger]


model.compile(optimizer=Adam(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=["accuracy", mean_iou])
history = model.fit_generator(dataloader.trainData(),
                              steps_per_epoch=200,
                              epochs=60,
                              validation_steps=10,
                              validation_data=dataloader.validData(),
                              callbacks=callbacks)
model.save_weights(log_dir + '\\weight.h5')
