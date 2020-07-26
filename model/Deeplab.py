
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras import Input
from keras.initializers import glorot_uniform
from keras.utils import plot_model
print(tf.__version__)


class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """

    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        outputs = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs * labels, 2),
                            1, keepdims=True)  # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)
        return point_score + trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # 目标y_pred需要是one hot形式
        if self.ignore_last_label:
            mask = 1 - y_true[:, :, -1:]
        else:
            mask = K.ones_like(y_pred[:, :, :1])
        y_true, y_pred = y_true[:, :,
                                :self.num_labels], y_pred[:, :, :self.num_labels]
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        init_states = [y_pred[:, 0]]  # 初始状态
        y_pred = K.concatenate([y_pred, mask])
        log_norm, _, _ = K.rnn(
            self.log_norm_step, y_pred[:, 1:], init_states)  # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        return log_norm - path_score  # 即log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1 - y_true[:, :, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :,
                                :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)


def identity_block(x, filters, dilation, BN=False):
    filters_1, filters_2, filters_3 = filters
    x_shortcut = x
    x = Conv2D(filters=filters_1, kernel_size=1, strides=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters_2, kernel_size=1, strides=(1, 1), padding='same',
               dilation_rate=dilation, kernel_initializer=glorot_uniform(seed=0))(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    # stage 3
    x = Conv2D(filters=filters_3, kernel_size=1, strides=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    # stage 4
    x = Add()([x, x_shortcut])
    x = Activation(activation='relu')(x)

    return x


def conv_block(x, filters, strides, dilation, BN=False):
    filters_1, filters_2, filters_3 = filters
    x_shortcut = x
    # stage 1
    x = Conv2D(filters=filters_1, kernel_size=1, strides=1, padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filters=filters_2, kernel_size=3, strides=strides, padding='same',
               dilation_rate=dilation, kernel_initializer=glorot_uniform(seed=0))(x)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters_3, kernel_size=1, strides=1, activation='relu', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    # stage 4
    x_shortcut = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=strides, padding='same',
                        kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
    # stage 5
    x = Add()([x, x_shortcut])
    x = Activation(activation='relu')(x)
    return x


def _resnet101(input_shape, BN=False):
    inputs = Input(input_shape)

    # 特征图大小下降一半
    x = Conv2D(filters=64, kernel_size=7, strides=2,
               padding='SAME', use_bias=False)(inputs)
    if BN:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    x = conv_block(x, filters=[64, 64, 256], strides=1, dilation=1)
    x = identity_block(x, filters=[64, 64, 256], dilation=1)
    x = identity_block(x, filters=[64, 64, 256], dilation=1)

    x = conv_block(x, filters=[128, 128, 512], strides=2, dilation=1)
    x = identity_block(x, filters=[128, 128, 512], dilation=1)
    x = identity_block(x, filters=[128, 128, 512],  dilation=1)
    x = identity_block(x, filters=[128, 128, 512], dilation=1)

    x = conv_block(x,  filters=[256, 256, 1024], strides=1, dilation=2)
    for i in range(22):
        x = identity_block(x, filters=[256, 256, 1024], dilation=2)

    # x = identity_block(x, filters=[256, 256, 1024], dilation=2)
    # x = identity_block(x, filters=[256, 256, 1024],  dilation=2)
    # x = identity_block(x, filters=[256, 256, 1024], dilation=2)
    # x = identity_block(x,  filters=[256, 256, 1024],  dilation=2)
    # x = identity_block(x,  filters=[256, 256, 1024],  dilation=2)

    x = conv_block(x, filters=[512, 512, 2048], strides=1, dilation=4)
    x = identity_block(x, filters=[256, 256, 2048], dilation=4)
    x = identity_block(x, filters=[256, 256, 2048], dilation=4)

    return inputs, x


def my_upsampling(x, resize_factor=8, method=0):
    # 0：双线性插值。1：最近邻居法。2：双三次插值法。3：面积插值法
    # NHWC
    img_h = x.shape[1] * resize_factor
    img_w = x.shape[2] * resize_factor
    return tf.image.resize_images(x, (img_h, img_w), 0)


def build_deeplab(input_shape, num_classes, BN=False):

    inputs, x = _resnet101(input_shape, BN)
    # hole = 6
    b1 = Conv2D(filters=1024, kernel_size=3, strides=1,
                padding='SAME', activation='relu', dilation_rate=6)(x)
    b1 = Conv2D(filters=1024, kernel_size=1, strides=1, activation='relu')(b1)
    b1 = Conv2D(filters=num_classes, kernel_size=1,
                strides=1, activation='relu')(b1)

    # hole = 12
    b2 = Conv2D(filters=1024, kernel_size=3, strides=1,
                padding='SAME', activation='relu', dilation_rate=12)(x)
    b2 = Conv2D(filters=1024, kernel_size=1, strides=1, activation='relu')(b2)
    b2 = Conv2D(filters=num_classes, kernel_size=1,
                strides=1, activation='relu')(b2)

    # hole = 18
    b3 = Conv2D(filters=1024, kernel_size=3, strides=1,
                padding='SAME', activation='relu', dilation_rate=18)(x)
    b3 = Conv2D(filters=1024, kernel_size=1, strides=1, activation='relu')(b3)
    b3 = Conv2D(filters=num_classes, kernel_size=1,
                strides=1, activation='relu')(b3)

    # hole = 24
    b4 = Conv2D(filters=1024, kernel_size=3, strides=1,
                padding='SAME', activation='relu', dilation_rate=24)(x)
    b4 = Conv2D(filters=1024, kernel_size=1, strides=1, activation='relu')(b4)
    b4 = Conv2D(filters=num_classes, kernel_size=1,
                strides=1, activation='relu')(b4)

    s = Add()([b1, b2, b3, b4])
    outputs = Lambda(my_upsampling)(s)
    model = Model(inputs, outputs)

    return model


def Deeplab(input_shape, num_classes, BN=False):
    model = build_deeplab(input_shape, num_classes, BN)
    model.name = 'Deeplab'
    model.summary()
    return model


model = Deeplab((512, 512, 3), 21)
# plot_model(model, to_file='Deeplab.png', show_shapes=True)
