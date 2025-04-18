from typing import Tuple, List
from keras import Model, layers, Sequential
from keras.src.regularizers.regularizers import L1L2 as reg_L1L2
from keras.src.initializers import GlorotNormal
from tensorflow import random, exp, shape


class SamplingColorizer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = random.normal(shape=shape(z_mean))
        return z_mean + exp(0.5 * z_log_var) * epsilon


class EncoderColorizer(Model):

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (128, 128, 1),
            L1=0.0,
            L2=0.0,
            z_dim=64,
            **kwargs):

        super().__init__(**kwargs)

        self._l1 = L1
        self._l2 = L2
        self._shape = input_shape

        self.enc_1 = self._encoder_block(128, (3, 3), True)  # 64
        self.enc_2 = self._encoder_block(128, (3, 3), True)  # 32
        self.enc_3 = self._encoder_block(256, (3, 3), True)  # 16
        self.enc_4 = self._encoder_block(512, (3, 3), True)  # 8
        self.enc_5 = self._encoder_block(1024, (3, 3), True)  # 4
        self.enc_6 = self._encoder_block(2048, (3, 3), True)  # 2
        self.enc_7 = self._encoder_block(2048, (3, 3), True)  # 1

        self.enc_flatten = layers.Flatten()
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_log_var = layers.Dense(z_dim, name='z_log_var')

    def _encoder_block(
            self,
            filters: int,
            kernel_size: Tuple[int, int],
            apply_batch_norm=True):

        block = Sequential()
        block.add(layers.Conv2D(
            filters, kernel_size,
            padding='same', strides=2,
            kernel_regularizer=reg_L1L2(l1=self._l1, l2=self._l2)))

        if apply_batch_norm:
            block.add(layers.BatchNormalization())

        block.add(layers.LeakyReLU())
        return block

    def call(self, inputs, training=None):
        skip_conncetion = [inputs]
        skip_conncetion.append(self.enc_1(inputs))
        skip_conncetion.append(self.enc_2(skip_conncetion[-1]))
        skip_conncetion.append(self.enc_3(skip_conncetion[-1]))
        skip_conncetion.append(self.enc_4(skip_conncetion[-1]))
        skip_conncetion.append(self.enc_5(skip_conncetion[-1]))
        skip_conncetion.append(self.enc_6(skip_conncetion[-1]))
        x = self.enc_7(skip_conncetion[-1])

        flattened = self.enc_flatten(x)
        z_mean = self.z_mean(flattened)
        z_log_var = self.z_log_var(flattened)
        return z_mean, z_log_var, skip_conncetion

    def model(self):
        x = layers.Input(shape=(128, 128, 1))
        return Model(inputs=[x], outputs=self.call(x))


class DecoderColorizer(Model):

    def __init__(
            self,
            input_shape: Tuple[int] = (64,),
            # skip_conncetion: List[Sequential] = [],
            L1=0.0,
            L2=0.0,
            **kwargs):

        super().__init__(**kwargs)
        self._l1 = L1
        self._l2 = L2
        self._shape = input_shape

        self.dec_projection = layers.Dense(2048 * 1 * 1)
        self.dec_reshape = layers.Reshape((1, 1, 2048))
        self.dec_7 = self._decoder_block(2048, (3, 3), False)  # 2
        self.dec_6 = self._decoder_block(1024, (3, 3), False)  # 4
        self.dec_5 = self._decoder_block(512, (3, 3), False)  # 8
        self.dec_4 = self._decoder_block(256, (3, 3), False)  # 16
        self.dec_3 = self._decoder_block(128, (3, 3), False)  # 32
        self.dec_2 = self._decoder_block(128, (3, 3), False)  # 64
        self.dec_1 = self._decoder_block(2, (3, 3), False)  # 128

        self.final_conv = layers.Conv2D(
            2, (3, 3),
            padding='same',
            strides=1,
            activation='tanh',
            kernel_initializer=GlorotNormal()
        )

    def _decoder_block(
            self,
            filters: int,
            kernel_size: Tuple[int, int],
            apply_dropout=True,
            dropout_rate=0.5):

        block = Sequential()
        block.add(layers.Conv2DTranspose(
            filters, kernel_size,
            padding='same', strides=2,
            kernel_regularizer=reg_L1L2(l1=self._l1, l2=self._l2)))

        if apply_dropout:
            block.add(layers.Dropout(dropout_rate))

        block.add(layers.LeakyReLU())
        return block

    def call(self, sample, skip_conncetion: List[Sequential], training=None):

        projection = self.dec_projection(sample)
        reshaped = self.dec_reshape(projection)

        y7 = self.dec_7(reshaped)
        y7 = layers.concatenate([y7, skip_conncetion.pop()], axis=-1)

        y6 = self.dec_6(y7)
        y6 = layers.concatenate([y6, skip_conncetion.pop()], axis=-1)

        y5 = self.dec_5(y6)
        y5 = layers.concatenate([y5, skip_conncetion.pop()], axis=-1)

        y4 = self.dec_4(y5)
        y4 = layers.concatenate([y4, skip_conncetion.pop()], axis=-1)

        y3 = self.dec_3(y4)
        y3 = layers.concatenate([y3, skip_conncetion.pop()], axis=-1)

        y2 = self.dec_2(y3)
        y2 = layers.concatenate([y2, skip_conncetion.pop()], axis=-1)

        y1 = self.dec_1(y2)
        y1 = layers.concatenate([y1, skip_conncetion.pop()], axis=-1)

        return self.final_conv(y1)

    def model(self):
        x = layers.Input(shape=(64,))
        return Model(inputs=[x], outputs=self.call(x))


class Colorizer(Model):

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (128, 128, 1),
            L1=0.0,
            L2=0.0,
            z_dim=64,
            **kwargs):

        super().__init__(**kwargs)
        self._l1 = L1
        self._l2 = L2
        self._shape = input_shape

        self.encoder = EncoderColorizer(input_shape, L1, L2, z_dim)

        self.sampling = SamplingColorizer()

        self.decoder = DecoderColorizer((z_dim,), L1, L2)

    def call(self, inputs, training=None):

        z_mean, z_log_var, skip_conn = self.encoder(inputs)
        sample = self.sampling([z_mean, z_log_var])
        outputs = self.decoder(sample, skip_conn)
        return outputs

    def model(self):
        x = layers.Input(shape=self._shape)
        return Model(inputs=[x], outputs=self.call(x))
