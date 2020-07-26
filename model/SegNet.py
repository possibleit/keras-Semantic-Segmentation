from keras.engine import Layer
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization,  MaxPool2D, Dense, Softmax, UpSampling2D, Reshape
from keras import Input, Model
from keras.utils import plot_model



class MaxPoolingWithArgmax2D(Layer):
    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                # output, argmax=K.tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, up_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.up_size = up_size

    def call(self, inputs, output_shape=None):

        updates, mask = inputs[0], inputs[1]
        with tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.up_size[0],
                    input_shape[2] * self.up_size[1],
                    input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.up_size[0],
            mask_shape[2] * self.up_size[1],
            mask_shape[3]
        )


def build_vgg16(input_shape):
    inputs = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='block1_conv1', kernel_initializer='he_normal')(inputs)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # 208,208,64 -> 128,128,128
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv1', kernel_initializer='he_normal')(x[0])
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # 104,104,128 -> 52,52,256
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv1', kernel_initializer='he_normal')(x[0])
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv3', kernel_initializer='he_normal')(x)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # 52,52,256 -> 26,26,512
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv1', kernel_initializer='he_normal')(x[0])
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv3', kernel_initializer='he_normal')(x)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # 26,26,512 -> 13,13,512
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv1', kernel_initializer='he_normal')(x[0])
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv3', kernel_initializer='he_normal')(x)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    return inputs, [f1, f2, f3, f4, f5]


def decoder(num_class, features):

    o = MaxUnpooling2D((2, 2))(features[-1])
    o = Conv2D(filters=512, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = MaxUnpooling2D((2, 2))([o, features[-2][1]])
    o = Conv2D(filters=512, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = Conv2D(filters=512, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = Conv2D(filters=256, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = MaxUnpooling2D((2, 2))([o, features[-3][1]])
    o = Conv2D(filters=256, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = Conv2D(filters=256, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = Conv2D(filters=128, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = MaxUnpooling2D((2, 2))([o, features[-4][1]])
    o = Conv2D(filters=128, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = Conv2D(filters=64, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = MaxUnpooling2D((2, 2))([o, features[-5][1]])
    o = Conv2D(filters=64, kernel_size=3, activation='relu',
               padding='SAME', kernel_initializer='he_normal')(o)
    o = Conv2D(filters=num_class, kernel_size=1, activation='softmax',
               padding='SAME', kernel_initializer='he_normal')(o)

    return o


def build_segnet(input_shape, num_class):
    inputs, features = build_vgg16(input_shape)
    outputs = decoder(num_class, features)
    model = Model(inputs, outputs)

    return model


def SegNet(input_shape, num_class=2):
    model = build_segnet(input_shape, num_class)
    model.name = "SegNet"
    model.summary()
    return model


# model = SegNet((224, 224, 3), 21)
# plot_model(model, to_file='SegNet.png', show_shapes=True)
