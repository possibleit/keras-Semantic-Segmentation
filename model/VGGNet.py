
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras import Model, Input
from keras.utils import plot_model


def build_vggnet(input_shape, include_top=False):
    inputs = Input(input_shape)
    o = Conv2D(64, kernel_size=3, padding='SAME',
               activation='relu', name='block1_conv1')(inputs)
    o = Conv2D(64, kernel_size=3, padding='SAME',
               activation='relu', name='block1_conv2')(o)
    o = MaxPool2D(pool_size=2, strides=2, name='block1maxpool')(o)

    o = Conv2D(128, kernel_size=3, padding='SAME',
               activation='relu', name='block2_conv1')(o)
    o = Conv2D(128, kernel_size=3, padding='SAME',
               activation='relu', name='block2_conv2')(o)
    o = MaxPool2D(pool_size=2, strides=2, name='block2maxpool')(o)

    o = Conv2D(256, kernel_size=3, padding='SAME',
               activation='relu', name='block3_conv1')(o)
    o = Conv2D(256, kernel_size=3, padding='SAME',
               activation='relu', name='block3_conv2')(o)
    o = Conv2D(256, kernel_size=3, padding='SAME',
               activation='relu', name='block3_conv3')(o)
    o = MaxPool2D(pool_size=2, strides=2, name='block3maxpool')(o)

    o = Conv2D(512, kernel_size=3, padding='SAME',
               activation='relu', name='block4_conv1')(o)
    o = Conv2D(512, kernel_size=3, padding='SAME',
               activation='relu', name='block4_conv2')(o)
    o = Conv2D(512, kernel_size=3, padding='SAME',
               activation='relu', name='block4_conv3')(o)
    o = MaxPool2D(pool_size=2, strides=2, name='block4maxpool')(o)

    o = Conv2D(512, kernel_size=3, padding='SAME',
               activation='relu', name='block5_conv1')(o)
    o = Conv2D(512, kernel_size=3, padding='SAME',
               activation='relu', name='block5_conv2')(o)
    o = Conv2D(512, kernel_size=3, padding='SAME',
               activation='relu', name='block5_conv3')(o)
    o = MaxPool2D(pool_size=2, strides=2, name='block5maxpool')(o)

    if include_top:
        o = Flatten(name='faltten')(o)
        o = Dense(4096, activation='relu', name='FC1')(o)
        o = Dense(4096, activation='relu', name='FC2')(o)
        o = Dense(1000, activation='softmax', name='SoftMax')(o)

    model = Model(inputs=inputs, outputs=o)

    return model


def VGGNet(input_shape, include_top=False):
    model = build_vggnet(input_shape, include_top)
    model.name = 'vgg16'
    model.summary()

    return model


model = VGGNet((224, 224, 3), include_top=True)
plot_model(model, to_file='VGGNet.png', show_shapes=True)
