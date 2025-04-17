
from keras import Model, layers, Sequential
from keras.src.regularizers.regularizers import L1L2 as reg_L1L2
from keras.src.initializers import GlorotNormal
from typing import Tuple


class Colorizer(Model):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (128, 128, 1),
                 L1=0.0,
                 L2=0.0):
        super().__init__()
        self._l1 = L1
        self._l2 = L2
        self._shape = input_shape

        self.enc_1 = self._encoder_block(128, (3, 3), False)
        self.enc_2 = self._encoder_block(128, (3, 3), False)
        self.enc_3 = self._encoder_block(256, (3, 3), True)
        self.enc_4 = self._encoder_block(512, (3, 3), True)
        self.enc_5 = self._encoder_block(1024, (3, 3), True)
        self.enc_6 = self._encoder_block(2048, (3, 3), True)

        self.latent = self._encoder_block(2048, (3, 3), True)

        self.dec_6 = self._decoder_block(2048, (3, 3), False)
        self.dec_5 = self._decoder_block(1024, (3, 3), False)
        self.dec_4 = self._decoder_block(512, (3, 3), False)
        self.dec_3 = self._decoder_block(256, (3, 3), False)
        self.dec_2 = self._decoder_block(128, (3, 3), False)
        self.dec_1 = self._decoder_block(128, (3, 3), False)
        self.dec_out = self._decoder_block(2, (3, 3), False)

        self.final_conv = layers.Conv2D(
            2, (3, 3),
            padding='same',
            strides=1,
            activation='tanh',
            kernel_initializer=GlorotNormal()
        )

        return self

    def _encoder_block(self,
                       filters: int,
                       kernel_size: Tuple[int, int],
                       apply_batch_norm=True):

        block = Sequential()
        block.add(layers.Conv2D(filters,
                                kernel_size,
                                padding='same',
                                strides=2,
                                kernel_regularizer=reg_L1L2(
                                    l1=self._l1,
                                    l2=self._l2)))

        if apply_batch_norm:
            block.add(layers.BatchNormalization())

        block.add(layers.LeakyReLU())
        return block

    def _decoder_block(self,
                       filters: int,
                       kernel_size: Tuple[int, int],
                       apply_dropout=False,
                       dropout_rate=0.5):

        block = Sequential()
        block.add(layers.Conv2DTranspose(filters,
                                         kernel_size,
                                         padding='same',
                                         strides=2,
                                         kernel_regularizer=reg_L1L2(
                                             l1=self._l1,
                                             l2=self._l2)))

        if apply_dropout:
            block.add(layers.Dropout(dropout_rate))

        block.add(layers.LeakyReLU())
        return block

    def call(self, inputs, training=None):
        x1 = self.enc_1(inputs)
        x2 = self.enc_2(x1)
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        x5 = self.enc_5(x4)
        x6 = self.enc_6(x5)

        latent = self.latent(x6)

        y6 = self.dec_6(latent)
        y6 = layers.concatenate([y6, x6], axis=-1)

        y5 = self.dec_5(y6)
        y5 = layers.concatenate([y5, x5], axis=-1)

        y4 = self.dec_4(y5)
        y4 = layers.concatenate([y4, x4], axis=-1)

        y3 = self.dec_3(y4)
        y3 = layers.concatenate([y3, x3], axis=-1)

        y2 = self.dec_2(y3)
        y2 = layers.concatenate([y2, x2], axis=-1)

        y1 = self.dec_1(y2)
        y1 = layers.concatenate([y1, x1], axis=-1)

        out = self.dec_out(y1)
        out = layers.concatenate([out, inputs], axis=-1)

        return self.final_conv(out)

    def model(self):
        x = layers.Input(shape=self._shape)
        return Model(inputs=[x], outputs=self.call(x))
