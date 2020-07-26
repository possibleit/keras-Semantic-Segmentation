from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.core import SpatialDropout2D, Permute, Activation, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Input
from keras.models import Model
from keras.utils import plot_model


def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same',
                  strides=strides, kernel_initializer='he_normal')(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=3)
    return merged


def bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    internal = output // internal_scale
    encoder = inp
    # 1x1
    # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    input_stride = 2 if downsample else 1
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False, kernel_initializer='he_normal')(encoder)
    # Batch normalization + PReLU
    # enet uses momentum of 0.1, keras default is 0.99
    encoder = BatchNormalization(momentum=0.1)(encoder)
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same',
                         kernel_initializer='he_normal')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same',
                         use_bias=False, kernel_initializer='he_normal')(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same',
                         kernel_initializer='he_normal')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (3, 3), dilation_rate=(
            dilated, dilated), padding='same', kernel_initializer='he_normal')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    # enet uses momentum of 0.1, keras default is 0.99
    encoder = BatchNormalization(momentum=0.1)(encoder)
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False,
                     kernel_initializer='he_normal')(encoder)

    # enet uses momentum of 0.1, keras default is 0.99
    encoder = BatchNormalization(momentum=0.1)(encoder)
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder


def en_build(inp, dropout_rate=0.01):
    enet = initial_block(inp)
    # enet_unpooling uses momentum of 0.1, keras default is 0.99
    enet = BatchNormalization(momentum=0.1)(enet)
    enet = PReLU(shared_axes=[1, 2])(enet)
    enet = bottleneck(enet, 64, downsample=True,
                      dropout_rate=dropout_rate)  # bottleneck 1.0
    for _ in range(4):
        # bottleneck 1.i
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate)

    enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for _ in range(2):
        enet = bottleneck(enet, 128)  # bottleneck 2.1
        enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = bottleneck(enet, 128)  # bottleneck 2.5
        enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8

    return enet

# decoder


def de_bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False,
               kernel_initializer='he_normal')(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True,
                   kernel_initializer='he_normal')(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(
            2, 2), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal')(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same',
                       use_bias=False, kernel_initializer='he_normal')(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder


def de_build(encoder, nc):
    enet = de_bottleneck(encoder, 64, upsample=True,
                         reverse_module=True)  # bottleneck 4.0
    enet = de_bottleneck(enet, 64)  # bottleneck 4.1
    enet = de_bottleneck(enet, 64)  # bottleneck 4.2
    enet = de_bottleneck(enet, 16, upsample=True,
                         reverse_module=True)  # bottleneck 5.0
    enet = de_bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(
        2, 2), strides=(2, 2), padding='same')(enet)
    return enet


def ENet(input_shape, n_classes):

    img_input = Input(shape=input_shape)
    enet = en_build(img_input)
    enet = de_build(enet, n_classes)
    o_shape = Model(img_input, enet).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    enet = (Reshape((outputHeight*outputWidth, n_classes)))(enet)
    enet = Activation('softmax')(enet)
    model = Model(img_input, enet)
    model.name = 'ENet'

    return model


model = ENet((512, 512, 3), n_classes=12)
model.summary()
# plot_model(model, to_file='ENet.png', show_shapes=True)
