import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, Conv3D, RNN, GRUCell, StackedRNNCells, ReLU, \
    Reshape, BatchNormalization, ReLU, Dropout, MaxPooling1D, Dropout, TimeDistributed, LSTM, ConvLSTM2D, add
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from data_loader import load_data
from metrics import TSS


class CNN_LSTM_(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self, input_shape):
        self.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape)))
        self.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        self.add(TimeDistributed(Dropout(0.5)))
        self.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.add(TimeDistributed(Flatten()))
        self.add(LSTM(100))
        self.add(Dropout(0.5))
        self.add(Dense(100, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))


class DeepSense(tf.keras.Model):

    def __init__(self, input_shape=(None, 8, 20, 3, 2), output_bias=None):
        super(DeepSense, self).__init__()

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        self.acc_conv1 = Conv3D(64, kernel_size=(3, 3, 3), activation=None, padding='valid',
                                input_shape=input_shape)
        self.acc_batch_norm1 = BatchNormalization()
        self.acc_act1 = ReLU()
        self.acc_dropout1 = Dropout(0.5)

        self.acc_conv2 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.acc_batch_norm2 = BatchNormalization()
        self.acc_act2 = ReLU()
        self.acc_dropout2 = Dropout(0.5)

        self.acc_conv3 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.acc_batch_norm3 = BatchNormalization()
        self.acc_act3 = ReLU()

        self.acc_shortcut = Conv3D(64, kernel_size=(3, 3, 3), activation=None, padding='valid')

        self.gyro_conv1 = Conv3D(64, kernel_size=(3, 3, 3), activation=None, padding='valid',
                                 input_shape=input_shape)
        self.gyro_batch_norm1 = BatchNormalization()
        self.gyro_act1 = ReLU()
        self.gyro_dropout1 = Dropout(0.5)

        self.gyro_conv2 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gyro_batch_norm2 = BatchNormalization()
        self.gyro_act2 = ReLU()
        self.gyro_dropout2 = Dropout(0.5)

        self.gyro_conv3 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gyro_batch_norm3 = BatchNormalization()
        self.gyro_act3 = ReLU()

        self.gyro_shortcut = Conv3D(64, kernel_size=(3, 3, 3), activation=None, padding='valid')

        self.gps_conv1 = Conv3D(64, kernel_size=(3, 3, 2), activation=None, padding='valid')
        self.gps_batch_norm1 = BatchNormalization()
        self.gps_act1 = ReLU()
        self.gps_dropout1 = Dropout(0.5)

        self.gps_conv2 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gps_batch_norm2 = BatchNormalization()
        self.gps_act2 = ReLU()
        self.gps_dropout2 = Dropout(0.5)

        self.gps_conv3 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gps_batch_norm3 = BatchNormalization()
        self.gps_act3 = ReLU()
        self.gps_dropout3 = Dropout(0.5)

        self.gps_shortcut = Conv3D(64, kernel_size=(3, 3, 2), activation=None, padding='valid')

        self.sensor_dropout = Dropout(0.5)

        self.sensor_conv1 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm1 = BatchNormalization()
        self.sensor_act1 = ReLU()
        self.sensor_dropout1 = Dropout(0.5)

        self.sensor_conv2 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm2 = BatchNormalization()
        self.sensor_act2 = ReLU()
        self.sensor_dropout2 = Dropout(0.5)

        self.sensor_conv3 = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm3 = BatchNormalization()
        self.sensor_act3 = ReLU()
        self.sensor_dropout3 = Dropout(0.5)

        self.sensor_shortcut = Conv3D(64, kernel_size=(3, 3, 1), activation=None, padding='same')

        self.sensor_reshape = Reshape((input_shape[2] - 2, (input_shape[1] - 2) * 3 * 64))

        self.sensor_gru1 = GRUCell(120, activation=None)
        self.sensor_gru2 = GRUCell(120, activation=None)
        self.sensor_stacked_rnn = RNN(StackedRNNCells([self.sensor_gru1, self.sensor_gru2]), return_sequences=True)

        self.sensor_gru1_dropout = GRUCell(120, dropout=0.5, activation=None)
        self.sensor_gru2_dropout = GRUCell(120, dropout=0.5, activation=None)
        self.sensor_stacked_rnn_dropout = RNN(StackedRNNCells([self.sensor_gru1_dropout, self.sensor_gru2_dropout]),
                                              return_sequences=True)

        self.fc = Dense(1, activation='sigmoid', bias_initializer=output_bias)

    def call(self, x, training):
        # split sensors
        acc, gyro, gps = tf.split(x, num_or_size_splits=3, axis=3)

        # remove incidents
        gps = gps[:, :, :, :2]

        # split real and imaginary part of complex accelerometer data
        acc_real = tf.math.real(acc)
        acc_imag = tf.math.imag(acc)
        acc = tf.stack((acc_real, acc_imag), axis=4)

        # split real and imaginary part of complex gyrosensor data
        gyro_real = tf.math.real(gyro)
        gyro_imag = tf.math.imag(gyro)
        gyro = tf.stack((gyro_real, gyro_imag), axis=4)

        # get real part of complex gps data
        gps = tf.math.real(gps)

        acc_conv1 = self.acc_conv1(acc)
        acc_conv1 = self.acc_batch_norm1(acc_conv1)
        acc_conv1 = self.acc_act1(acc_conv1)
        acc_conv1 = self.acc_dropout1(acc_conv1) if training else acc_conv1

        acc_conv2 = self.acc_conv2(acc_conv1)
        acc_conv2 = self.acc_batch_norm2(acc_conv2)
        acc_conv2 = self.acc_act2(acc_conv2)
        acc_conv2 = self.acc_dropout2(acc_conv2) if training else acc_conv2

        acc_conv3 = self.acc_conv3(acc_conv2)
        acc_conv3 = self.acc_batch_norm3(acc_conv3)
        acc_conv3 = self.acc_act3(acc_conv3)

        acc_shortcut = self.acc_shortcut(acc)
        acc_shortcut = self.acc_batch_norm3(acc_shortcut)
        acc_shortcut = self.acc_act3(acc_shortcut)

        acc = add([acc_conv3, acc_shortcut])

        gyro_conv1 = self.gyro_conv1(gyro)
        gyro_conv1 = self.gyro_batch_norm1(gyro_conv1)
        gyro_conv1 = self.gyro_act1(gyro_conv1)
        gyro_conv1 = self.gyro_dropout1(gyro_conv1) if training else gyro_conv1

        gyro_conv2 = self.gyro_conv2(gyro_conv1)
        gyro_conv2 = self.gyro_batch_norm2(gyro_conv2)
        gyro_conv2 = self.gyro_act2(gyro_conv2)
        gyro_conv2 = self.gyro_dropout2(gyro_conv2) if training else gyro_conv2

        gyro_conv3 = self.gyro_conv3(gyro_conv2)
        gyro_conv3 = self.gyro_batch_norm3(gyro_conv3)
        gyro_conv3 = self.gyro_act3(gyro_conv3)

        gyro_shortcut = self.gyro_shortcut(gyro)
        gyro_shortcut = self.gyro_batch_norm3(gyro_shortcut)
        gyro_shortcut = self.gyro_act3(gyro_shortcut)

        gyro = add([gyro_conv3, gyro_shortcut])

        gps = gps[:, :, :, :, tf.newaxis]

        gps_conv1 = self.gps_conv1(gps)
        gps_conv1 = self.gps_batch_norm1(gps_conv1)
        gps_conv1 = self.gps_act1(gps_conv1)
        gps_conv1 = self.gps_dropout1(gps_conv1) if training else gps_conv1

        gps_conv2 = self.gps_conv2(gps_conv1)
        gps_conv2 = self.gps_batch_norm2(gps_conv2)
        gps_conv2 = self.gps_act2(gps_conv2)
        gps_conv2 = self.gps_dropout2(gps_conv2) if training else gps_conv2

        gps_conv3 = self.gps_conv3(gps_conv2)
        gps_conv3 = self.gps_batch_norm3(gps_conv3)
        gps_conv3 = self.gps_act3(gps_conv3)

        gps_shortcut = self.gps_shortcut(gps)
        gps_shortcut = self.gps_batch_norm3(gps_shortcut)
        gps_shortcut = self.gps_act3(gps_shortcut)

        gps = add([gps_conv3, gps_shortcut])

        sensor = tf.concat([acc, gyro, gps], 3)

        sensor = self.sensor_dropout(sensor)

        sensor_conv1 = self.sensor_conv1(sensor)
        sensor_conv1 = self.sensor_batch_norm1(sensor_conv1)
        sensor_conv1 = self.sensor_act1(sensor_conv1)
        sensor_conv1 = self.sensor_dropout1(sensor_conv1) if training else sensor_conv1

        sensor_conv2 = self.sensor_conv2(sensor_conv1)
        sensor_conv2 = self.sensor_batch_norm2(sensor_conv2)
        sensor_conv2 = self.sensor_act2(sensor_conv2)
        sensor_conv2 = self.sensor_dropout2(sensor_conv2) if training else sensor_conv2

        sensor_conv3 = self.sensor_conv3(sensor_conv2)
        sensor_conv3 = self.sensor_batch_norm3(sensor_conv3)
        sensor_conv3 = self.sensor_act3(sensor_conv3)
        sensor_conv3 = self.sensor_dropout3(sensor_conv3) if training else sensor_conv3

        sensor_shortcut = self.sensor_shortcut(sensor)
        sensor_shortcut = self.sensor_batch_norm3(sensor_shortcut)
        sensor_shortcut = self.sensor_act3(sensor_shortcut)
        sensor_shortcut = self.sensor_dropout3(sensor_shortcut) if training else sensor_shortcut

        sensor = add([sensor_conv3, sensor_shortcut])

        sensor = tf.transpose(sensor, perm=(0, 2, 1, 3, 4))
        sensor = self.sensor_reshape(sensor)

        sensor = self.sensor_stacked_rnn_dropout(sensor) if training else self.sensor_stacked_rnn(sensor)

        sensor = tf.math.reduce_mean(sensor, axis=1, keepdims=False)

        sensor = self.fc(sensor)

        return sensor


def train(train_ds, val_ds, test_ds, class_weight, num_epochs=10, patience=1, input_shape=(None, 8, 20, 3, 2),
          deepsense=True, checkpoint_dir='checkpoints/cnn/training'):
    initial_bias = np.log(class_weight[0] / class_weight[1])

    if deepsense:
        model = DeepSense(input_shape, initial_bias)

    else:
        model = CNN_LSTM_()
        model.create_model(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.TrueNegatives(name='tn'), tf.keras.metrics.FalsePositives(name='fp'),
                           tf.keras.metrics.FalseNegatives(name='fn'), tf.keras.metrics.TruePositives(name='tp'),
                           tf.keras.metrics.AUC(curve='PR', from_logits=False), TSS(), tf.keras.metrics.SensitivityAtSpecificity(0.96, name='sas')
                           ])

    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
    try:
        model.load_weights(latest)
    except:
        print('There is no existing checkpoint')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        monitor='val_auc',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=True,
        save_freq='epoch')

    # Create a callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=patience,
        verbose=1,
        mode='max',
        restore_best_weights=True)

    # Define the Keras TensorBoard callback.
    tb_logdir = 'tb_logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=1)

    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs,
              callbacks=[cp_callback, es_callback, tensorboard_callback], class_weight=class_weight)

    print('Model evaluation on train set:')
    model.evaluate(train_ds)
    print('Model evaluation on val set:')
    model.evaluate(val_ds)
    print('Model evaluation on test set:')
    model.evaluate(test_ds)

    y_pred = model.predict(test_ds)

    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    y_pred = np.round(y_pred)[:, 0]

    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

    model.summary()


if __name__ == '__main__':
    dir = '../Ride_Data'
    checkpoint_dir = 'checkpoints/cnn/training'
    target_region = 'Berlin'
    bucket_size = 100
    batch_size = 128
    in_memory = False
    num_epochs = 100
    patience = 10
    deepsense = True
    fft_window = 8
    image_width = 20
    class_counts_file = os.path.join(dir, 'class_counts.csv')

    if deepsense:
        input_shape = (None, fft_window, image_width, 3, 2)
    else:
        input_shape = (None, 4, int(bucket_size / 4), 8)


    train_ds, val_ds, test_ds, class_weight = load_data(dir, target_region, input_shape, batch_size, in_memory, deepsense, class_counts_file)
    train(train_ds, val_ds, test_ds, class_weight, num_epochs, patience, input_shape, deepsense, checkpoint_dir)
