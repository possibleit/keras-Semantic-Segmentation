from keras import Model, Input
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from keras.utils import plot_model
# from MyLayer import MaxUnpooling2D, MaxPoolingWithArgmax2D


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


def buildConv2DBlock(block_input, filters, block, depth):
    for i in range(1, depth + 1):
        if i == 1:
            conv2d = Conv2D(filters, 3, padding='same', name='conv{}-{}'.format(block, i), use_bias=False, activation='relu', kernel_initializer='he_normal')(
                block_input)
        else:
            conv2d = Conv2D(filters, 3, padding='same', name='conv{}-{}'.format(block, i),
                            use_bias=False, activation='relu', kernel_initializer='he_normal')(conv2d)

    return conv2d


def build_deconvnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = buildConv2DBlock(inputs, 64, 1, 2)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='pool1')(x)
    f1 = x

    x = buildConv2DBlock(x[0], 128, 2, 2)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='pool2')(x)
    f2 = x

    x = buildConv2DBlock(x[0], 256, 3, 3)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='pool3')(x)
    f3 = x

    x = buildConv2DBlock(x[0], 512, 4, 3)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='pool4')(x)
    f4 = x

    x = buildConv2DBlock(x[0], 512, 5, 3)
    x = MaxPoolingWithArgmax2D((2, 2), strides=(2, 2), name='pool5')(x)
    f5 = x

    fc6 = Conv2D(512, 7, use_bias=False, padding='valid', activation='relu',
                 kernel_initializer='he_normal', name='fc6')(x[0])  # 4096

    fc7 = Conv2D(512, 1, use_bias=False, padding='valid', activation='relu',
                 kernel_initializer='he_normal', name='fc7')(fc6)  # 4096

    x = Conv2DTranspose(512, 7, use_bias=False, padding='valid', activation='relu',
                        kernel_initializer='he_normal', name='deconv-fc6')(fc7)

    x = MaxUnpooling2D((2, 2))([x, f5[1]])
    x = Conv2DTranspose(512, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv5-1')(x)

    x = Conv2DTranspose(512, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv5-2')(x)

    x = Conv2DTranspose(512, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv5-3')(x)

    x = MaxUnpooling2D((2, 2))([x, f4[1]])

    x = Conv2DTranspose(512, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv4-1')(x)

    x = Conv2DTranspose(512, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv4-2')(x)

    x = Conv2DTranspose(256, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv4-3')(x)

    x = MaxUnpooling2D((2, 2))([x, f3[1]])

    x = Conv2DTranspose(256, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv3-1')(x)

    x = Conv2DTranspose(256, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv3-2')(x)

    x = Conv2DTranspose(128, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv3-3')(x)

    x = MaxUnpooling2D((2, 2))([x, f2[1]])

    x = Conv2DTranspose(128, 3, use_bias=False, padding='same',
                        activation='relu', kernel_initializer='he_normal', name='deconv2-1')(x)

    x = Conv2DTranspose(64, 3, use_bias=False, padding='same', activation='relu',
                        kernel_initializer='he_normal', name='deconv2-2')(x)

    x = MaxUnpooling2D((2, 2))([x, f1[1]])

    x = Conv2DTranspose(64, 3, use_bias=False, padding='same', activation='relu',
                        kernel_initializer='he_normal', name='deconv1-1')(x)

    x = Conv2DTranspose(64, 3, use_bias=False, padding='same', activation='relu',
                        kernel_initializer='he_normal', name='deconv1-2')(x)

    outputs = Conv2DTranspose(
        num_classes, 1, activation='softmax', padding='same', name='output')(x)

    return inputs, outputs


def DeconvNet(input_shape, num_classes):
    inputs, outputs = build_deconvnet(input_shape, num_classes)
    model = Model(inputs=inputs, outputs=outputs)
    model.name = 'DeconvNet'
    return model


model = DeconvNet((512, 512, 3), 12)
model.summary()
# plot_model(model, to_file='DeconvNet.png', show_shapes=True)
