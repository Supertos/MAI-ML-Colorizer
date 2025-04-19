from typing import Tuple, List
from keras import Model, layers, Sequential
from keras.src.regularizers.regularizers import L1L2 as reg_L1L2
from keras.src.initializers import GlorotNormal
from tensorflow import random, exp, shape
from keras import ops, losses
from keras.api.metrics import Mean
from tensorflow import GradientTape


""" TODO:
    - Need to figure out how Keras draws model graphs,
    since it currently does not display layers hidden in classes.

    - + Add training (not trainable) for dropout in call()
    - + make encoder_layer class
    - + make own loss fun
    - With a large number of parameters, something strange happens with
    the loss function. Maybe it's normal????
"""


class SamplingColorizer(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = random.normal(shape=shape(z_mean))
        return z_mean + exp(0.5 * z_log_var) * epsilon  # tf.exp -> Kereas.exp?


class EncoderBlcok(layers.Layer):

    def __init__(
            self, filters: int, kernel_size: Tuple[int, int],
            trainable=True, L1=0.0, L2=0.0, **kwargs):

        super().__init__(trainable=trainable, **kwargs)

        self._conv2d = layers.Conv2D(
            filters, kernel_size,
            padding='same', strides=2,
            kernel_regularizer=reg_L1L2(l1=L1, l2=L2))
        self._batchNorm = layers.BatchNormalization()
        self._leakyReLU = layers.LeakyReLU()

    def call(self, inputs, training=False, *args, **kwargs):
        x = self._conv2d(inputs)
        x = self._batchNorm(inputs=x, training=training)
        x = self._leakyReLU(x)
        return x


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

        self.enc_1 = EncoderBlcok(32, (3, 3), True)  # 64
        self.enc_2 = EncoderBlcok(64, (3, 3), True)  # 32
        self.enc_3 = EncoderBlcok(128, (3, 3), True)  # 16
        self.enc_4 = EncoderBlcok(256, (3, 3), True)  # 8
        self.enc_5 = EncoderBlcok(512, (3, 3), True)  # 4
        # self.enc_6 = EncoderBlcok(1024, (3, 3), True)  # 2
        # self.enc_7 = EncoderBlcok(2048, (3, 3), True)  # 1

        self.enc_flatten = layers.Flatten()
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_log_var = layers.Dense(z_dim, name='z_log_var')

    def call(self, inputs, training=None):
        skip_connection = [inputs]
        skip_connection.append(self.enc_1(inputs, training=training))
        skip_connection.append(
            self.enc_2(
                skip_connection[-1],
                training=training))
        skip_connection.append(
            self.enc_3(
                skip_connection[-1],
                training=training))
        skip_connection.append(
            self.enc_4(
                skip_connection[-1],
                training=training))
        # skip_connection.append(
        #     self.enc_5(
        #         skip_connection[-1],
        #         training=training))
        # skip_connection.append(
        #     self.enc_6(
        #         skip_connection[-1],
        #         training=training))
        # x = self.enc_7(skip_connection[-1])
        x = self.enc_5(
                skip_connection[-1],
                training=training)

        flattened = self.enc_flatten(x)
        z_mean = self.z_mean(flattened)
        z_log_var = self.z_log_var(flattened)
        return z_mean, z_log_var, skip_connection

    def model(self):
        x = layers.Input(shape=(128, 128, 1))
        return Model(inputs=[x], outputs=self.call(x, training=False))


class DecoderBlcok(layers.Layer):

    def __init__(
            self, filters: int, kernel_size: Tuple[int, int],
            trainable=True, L1=0.0, L2=0.0, dropout_rate=0.5, **kwargs):

        super().__init__(trainable=trainable, **kwargs)

        self._conv2d = layers.Conv2DTranspose(
            filters, kernel_size,
            padding='same', strides=2,
            kernel_regularizer=reg_L1L2(l1=L1, l2=L2))
        self._dropout = layers.Dropout(dropout_rate)
        self._leakyReLU = layers.LeakyReLU()

    def call(self, inputs, skip_connection, training=False, *args, **kwargs):
        x = self._conv2d(inputs)
        x = self._dropout(inputs=x, training=training)
        x = self._leakyReLU(x)
        x = layers.concatenate([x, skip_connection], axis=-1)
        return x


class DecoderColorizer(Model):

    def __init__(
            self,
            input_shape: Tuple[int] = (64,),
            L1=0.0,
            L2=0.0,
            **kwargs):

        super().__init__(**kwargs)
        self._l1 = L1
        self._l2 = L2
        self._shape = input_shape

        self.dec_projection = layers.Dense(512 * 4 * 4)
        self.dec_reshape = layers.Reshape((4, 4, 512))
        # self.dec_7 = DecoderBlcok(1024, (3, 3), False)  # 2
        # self.dec_6 = DecoderBlcok(512, (3, 3), False)  # 4
        self.dec_5 = DecoderBlcok(256, (3, 3), False)  # 8
        self.dec_4 = DecoderBlcok(128, (3, 3), False)  # 16
        self.dec_3 = DecoderBlcok(64, (3, 3), False)  # 32
        self.dec_2 = DecoderBlcok(32, (3, 3), False)  # 64
        self.dec_1 = DecoderBlcok(2, (3, 3), False)  # 128

        self.final_conv = layers.Conv2D(
            2, (3, 3),
            padding='same',
            strides=1,
            activation='tanh',
            kernel_initializer=GlorotNormal()
        )

    def call(self, sample, skip_connection: List[Sequential], training=None):

        projection = self.dec_projection(sample)
        reshaped = self.dec_reshape(projection)

        # y7 = self.dec_7(reshaped, skip_connection.pop(), training=training)

        # y6 = self.dec_6(y7, skip_connection.pop(), training=training)

        # y5 = self.dec_5(y6, skip_connection.pop(), training=training)
        y5 = self.dec_5(reshaped, skip_connection.pop(), training=training)

        y4 = self.dec_4(y5, skip_connection.pop(), training=training)

        y3 = self.dec_3(y4, skip_connection.pop(), training=training)

        y2 = self.dec_2(y3, skip_connection.pop(), training=training)

        y1 = self.dec_1(y2, skip_connection.pop(), training=training)

        return self.final_conv(y1)

    def model(self):
        x = layers.Input(shape=(64,))
        return Model(
            inputs=[x], outputs=self.call(
                x, skip_connection=[], training=False))


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

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

        self.encoder = EncoderColorizer(input_shape, L1, L2, z_dim)

        self.sampling = SamplingColorizer()

        self.decoder = DecoderColorizer((z_dim,), L1, L2)

    def call(self, inputs, training=None):

        z_mean, z_log_var, skip_conn = self.encoder(inputs, training=training)
        sample = self.sampling([z_mean, z_log_var])
        outputs = self.decoder(sample, skip_conn, training=training)

        return outputs

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data

        with GradientTape() as tape:

            z_mean, z_log_var, skip_conn = self.encoder(x, training=True)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z, skip_conn, training=True)

            reconstruction_loss = ops.mean(
                ops.sum(
                    losses.mean_squared_error(y, reconstruction),
                    axis=(1, 2)  # уточнить
                )
            )

            kl_loss = -0.5 * ops.mean(
                ops.sum(
                    1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data

        z_mean, z_log_var, skip_conn = self.encoder(x, training=False)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z, skip_conn, training=False)

        reconstruction_loss = ops.mean(
            ops.sum(
                losses.mean_squared_error(y, reconstruction),
                axis=(1, 2)))
        kl_loss = -0.5 * ops.mean(
            ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                axis=1
            )
        )
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def model(self):
        x = layers.Input(shape=self._shape)
        return Model(inputs=[x], outputs=self.call(x, training=False))
