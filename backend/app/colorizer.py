from keras import layers
import keras
from typing import Tuple

SHAPE_ = (128, 128, 1)
L1_NORM = 1e-5 * 0.0
L2_NORM = 1e-6 * 0.0


def encoder_block(filters: int, kernel_size: Tuple[int, int],
                  apply_batch_normalization=True, l1_reg=0.0, l2_reg=0.0):
    downsample = keras.models.Sequential()
    downsample.add(layers.Conv2D(filters, kernel_size, padding='same', strides=2,
                                  kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample


def decoder_block(filters: int, kernel_size: Tuple[int, int], dropout=False,
                  l1_reg=0.0, l2_reg=0.0):
    upsample = keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size, padding='same', strides=2,
                                        kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    if dropout:
        upsample.add(layers.Dropout(0.2))
    upsample.add(keras.layers.LeakyReLU())
    return upsample


def build_colorizer(input_shape=SHAPE_, l2_reg=L2_NORM, l1_reg=L1_NORM):
    inputs = layers.Input(shape=input_shape)

    x1 = encoder_block(128, (3, 3), False)(inputs)
    x2 = encoder_block(128, (3, 3), False)(x1)
    x3 = encoder_block(256, (3, 3), True)(x2)
    x4 = encoder_block(512, (3, 3), True)(x3)
    x5 = encoder_block(1024, (3, 3), True)(x4)
    x6 = encoder_block(2048, (3, 3), True)(x5)

    b1 = encoder_block(2048, (3, 3), True)(x6)

    y6 = decoder_block(2048, (3, 3), False)(b1)
    y6 = layers.concatenate([y6, x6])

    y5 = decoder_block(1024, (3, 3), False)(y6)
    y5 = layers.concatenate([y5, x5])

    y4 = decoder_block(512, (3, 3), False)(y5)
    y4 = layers.concatenate([y4, x4])

    y3 = decoder_block(256, (3, 3), False)(y4)
    y3 = layers.concatenate([y3, x3])

    y2 = decoder_block(128, (3, 3), False)(y3)
    y2 = layers.concatenate([y2, x2])

    y1 = decoder_block(128, (3, 3), False)(y2)
    y1 = layers.concatenate([y1, x1])

    outputs = decoder_block(2, (3, 3), False)(y1)
    outputs = layers.concatenate([outputs, inputs])
    outputs = layers.Conv2D(
        2, (3, 3), padding='same', strides=1, activation='tanh',
        kernel_initializer=keras.initializers.GlorotNormal())(outputs)

    return keras.Model(inputs, outputs)
