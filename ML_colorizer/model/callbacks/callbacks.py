import tensorflow as tf
from cv2 import cvtColor, COLOR_Lab2BGR
import numpy as np
import shutil
import os
import time
import keras


class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, local_log_dir, val_data, max_images=3):
        super().__init__()

        self.file_writer = tf.summary.create_file_writer(
            local_log_dir + "/images2")
        self.val_data = val_data
        self.max_images = max_images
        self.local_log_dir = local_log_dir + "/images2"

    def on_epoch_end(self, epoch):

        x_batch, y_true_ab = next(iter(self.val_data))

        x_batch = x_batch[:self.max_images].numpy()
        y_true_ab = y_true_ab[:self.max_images].numpy()
        y_pred_ab = self.model.predict(x_batch)[:self.max_images]

        def lab_batch_to_rgb(L, ab):
            """
            L: float32 [H,W,1] в [0,1]
            ab: float32 [H,W,2] в [-1,1]
            """
            L = L[..., 0] * 255.0
            a = ab[..., 0] * 128.0
            b = ab[..., 1] * 128.0
            lab = np.stack([L, a, b], axis=-1)
            rgb = cvtColor(lab, COLOR_Lab2BGR)
            return rgb

        rgb_gray = []
        rgb_groundtruth = []
        rgb_predicted = []

        for i in range(x_batch.shape[0]):
            L = x_batch[i]            # [H,W,1]
            ab_t = y_true_ab[i]          # [H,W,2]
            ab_p = y_pred_ab[i]          # [H,W,2]

            gray_rgb = np.concatenate([L, L, L], axis=-1)
            rgb_gray.append(gray_rgb)

            rgb_groundtruth.append(lab_batch_to_rgb(L, ab_t))

            rgb_predicted.append(lab_batch_to_rgb(L, ab_p))

        rgb_gray = np.stack(rgb_gray, axis=0)
        rgb_groundtruth = np.stack(rgb_groundtruth, axis=0)
        rgb_predicted = np.stack(rgb_predicted, axis=0)

        with self.file_writer.as_default():
            tf.summary.image(
                "grayscale_L", rgb_gray, step=epoch,
                max_outputs=self.max_images)
            tf.summary.image(
                "ground_truth_ab", rgb_groundtruth, step=epoch,
                max_outputs=self.max_images)
            tf.summary.image(
                "prediction_ab", rgb_predicted, step=epoch,
                max_outputs=self.max_images)
        self.file_writer.flush()


class LayerHistograms(tf.keras.callbacks.Callback):
    def __init__(self, local_log_dir='/tmp/logs'):
        super().__init__()

        self.local_writer = tf.summary.create_file_writer(
            local_log_dir + "/layers")
        self.local_log_dir = local_log_dir + "/layers"

    def on_epoch_end(self, epoch, logs=None):
        with self.local_writer.as_default():
            for layer in self.model.layers:
                for weight in layer.trainable_weights:
                    tag = weight.path.replace(':', '_')
                    tf.summary.histogram(name=tag, data=weight, step=epoch)

        self.local_writer.flush()


class Cpy(tf.keras.callbacks.Callback):
    '''
    Нужен для копирования логов из локального хранилища
    в Google Colab в OneDrive, тк прямая запись на диск работает с нюансами
    '''

    def __init__(self, log_dir, local_dir):
        self.drive_log_dir = log_dir
        self.local_log_dir = local_dir

    def on_train_end(self):
        time.sleep(10)  # Логи могут не успеть записать в локальное хранилище
        os.makedirs(self.drive_log_dir, exist_ok=True)
        shutil.copytree(
            self.local_log_dir, self.drive_log_dir, dirs_exist_ok=True)


class GradientLoggingTensorBoard(keras.callbacks.TensorBoard):
    '''
    Расширение TB callback для получения гистограмм градентов
    на каждом из слоёв.
    Оригинальный TB совмещает все гистограммы в одну по типам kernal, bias etc.
    Пример использования:
    ```python
    tensorboard_cb2 = GradientLoggingTensorBoard(
    train_ds = train_ds,
    log_dir=log_dir_local,
    histogram_freq=1,
    write_graph=True
    )
    ```
    '''
    def __init__(self, train_ds, **kwargs):
        self.train_ds = train_ds
        super().__init__(**kwargs)

    def _log_gradients(self, epoch):
        x_sample, y_sample = next(iter(self.train_ds))
        with tf.GradientTape() as tape:
            preds = self.model(x_sample, training=True)
            loss = self.model.compiled_loss(y_sample, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)

        optimizer = self.model.optimizer
        if optimizer is not None:
            if (hasattr(optimizer, 'clipnorm')
                    and optimizer.clipnorm is not None):
                grads = [tf.clip_by_norm(g, optimizer.clipnorm) for g in grads]
            elif (hasattr(optimizer, 'clipvalue')
                    and optimizer.clipvalue is not None):
                grads = [
                    tf.clip_by_value(
                        g,
                        -optimizer.clipvalue,
                        optimizer.clipvalue)
                    for g in grads]
            elif (hasattr(optimizer, 'global_clipnorm')
                  and optimizer.global_clipnorm is not None):
                grads, _ = tf.clip_by_global_norm(
                    grads, optimizer.global_clipnorm)

        with self._train_writer.as_default():
            for var, grad in zip(self.model.trainable_variables, grads):
                if grad is not None:
                    tf.summary.histogram(
                        var.path.replace(':', '_') + '_grad',
                        grad,
                        step=epoch
                    )
        self._train_writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)
