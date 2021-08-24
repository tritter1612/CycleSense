import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten, Conv1D, MaxPooling1D, Dropout, TimeDistributed, LSTM, \
    ConvLSTM2D, Conv3D, BatchNormalization, ReLU, Reshape, GRUCell, RNN, StackedRNNCells
from tensorboard.plugins.hparams import api as hp

from data_loader import load_data
from metrics import TSS


class DeepSense(tf.keras.Model):

    def __init__(self, hparams, input_shape=(None, 8, 20, 3, 2)):
        super(DeepSense, self).__init__()

        self.acc_conv1 = Conv3D(hparams[HP_NUM_KERNELS_L1], kernel_size=(3, 3, 3), activation=None, padding='valid')
        self.acc_batch_norm1 = BatchNormalization()
        self.acc_act1 = ReLU()
        self.acc_dropout1 = Dropout(hparams[HP_DROPOUT_L1])

        self.acc_conv2 = Conv3D(hparams[HP_NUM_KERNELS_L2], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.acc_batch_norm2 = BatchNormalization()
        self.acc_act2 = ReLU()
        self.acc_dropout2 = Dropout(hparams[HP_DROPOUT_L2])

        self.acc_conv3 = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.acc_batch_norm3 = BatchNormalization()
        self.acc_act3 = ReLU()

        self.acc_shortcut = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 3), activation=None, padding='valid')

        self.gyro_conv1 = Conv3D(hparams[HP_NUM_KERNELS_L1], kernel_size=(3, 3, 3), activation=None, padding='valid')
        self.gyro_batch_norm1 = BatchNormalization()
        self.gyro_act1 = ReLU()
        self.gyro_dropout1 = Dropout(hparams[HP_DROPOUT_L3])

        self.gyro_conv2 = Conv3D(hparams[HP_NUM_KERNELS_L2], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gyro_batch_norm2 = BatchNormalization()
        self.gyro_act2 = ReLU()
        self.gyro_dropout2 = Dropout(hparams[HP_DROPOUT_L4])

        self.gyro_conv3 = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gyro_batch_norm3 = BatchNormalization()
        self.gyro_act3 = ReLU()

        self.gyro_shortcut = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 3), activation=None, padding='valid')

        self.gps_conv1 = Conv3D(hparams[HP_NUM_KERNELS_L1], kernel_size=(3, 3, 2), activation=None, padding='valid')
        self.gps_batch_norm1 = BatchNormalization()
        self.gps_act1 = ReLU()
        self.gps_dropout1 = Dropout(hparams[HP_DROPOUT_L5])

        self.gps_conv2 = Conv3D(hparams[HP_NUM_KERNELS_L2], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gps_batch_norm2 = BatchNormalization()
        self.gps_act2 = ReLU()
        self.gps_dropout2 = Dropout(hparams[HP_DROPOUT_L6])

        self.gps_conv3 = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gps_batch_norm3 = BatchNormalization()
        self.gps_act3 = ReLU()
        self.gps_dropout3 = Dropout(hparams[HP_DROPOUT_L7])

        self.gps_shortcut = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 2), activation=None, padding='valid')

        self.sensor_dropout = Dropout(hparams[HP_DROPOUT_L8])

        self.sensor_conv1 = Conv3D(hparams[HP_NUM_KERNELS_L4], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm1 = BatchNormalization()
        self.sensor_act1 = ReLU()
        self.sensor_dropout1 = Dropout(hparams[HP_DROPOUT_L9])

        self.sensor_conv2 = Conv3D(hparams[HP_NUM_KERNELS_L5], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm2 = BatchNormalization()
        self.sensor_act2 = ReLU()
        self.sensor_dropout2 = Dropout(hparams[HP_DROPOUT_L10])

        self.sensor_conv3 = Conv3D(hparams[HP_NUM_KERNELS_L6], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm3 = BatchNormalization()
        self.sensor_act3 = ReLU()
        self.sensor_dropout3 = Dropout(hparams[HP_DROPOUT_L11])

        self.sensor_shortcut = Conv3D(hparams[HP_NUM_KERNELS_L6], kernel_size=(3, 3, 1), activation=None, padding='same')

        self.sensor_reshape = Reshape((input_shape[2] - 2, (input_shape[1] - 2) *  3 * hparams[HP_NUM_KERNELS_L6]))

        if hparams[HP_RNN_CELL_TYPE] == 'stacked_RNN':

            self.sensor_gru1 = GRUCell(hparams[HP_RNN_UNITS], activation=None)
            self.sensor_gru2 = GRUCell(hparams[HP_RNN_UNITS], activation=None)
            self.sensor_rnn = RNN(StackedRNNCells([self.sensor_gru1, self.sensor_gru2]), return_sequences=True)

            self.sensor_gru1_dropout = GRUCell(hparams[HP_RNN_UNITS], dropout=hparams[HP_DROPOUT_L12], activation=None)
            self.sensor_gru2_dropout = GRUCell(hparams[HP_RNN_UNITS], dropout=hparams[HP_DROPOUT_L13], activation=None)
            self.sensor_rnn_dropout = RNN(StackedRNNCells([self.sensor_gru1_dropout, self.sensor_gru2_dropout]),
                                          return_sequences=True)

        elif hparams[HP_RNN_CELL_TYPE] == 'LSTM':

            self.sensor_rnn = LSTM(hparams[HP_RNN_UNITS], activation=None, return_sequences=True)
            self.sensor_rnn_dropout = LSTM(hparams[HP_RNN_UNITS], activation=None, return_sequences=True,
                                           dropout=hparams[HP_DROPOUT_L12])

        elif hparams[HP_RNN_CELL_TYPE] == 'None':

            self.sensor_rnn = Lambda((lambda x: x))
            self.sensor_rnn_dropout = Dropout(hparams[HP_DROPOUT_L12])

        self.flatten = Flatten()

        self.fc = Dense(1, activation='sigmoid')

    def call(self, x, training):
        # split sensors
        acc, gyro, gps = tf.split(x, num_or_size_splits=3, axis=3)

        # remove incidents
        gps = gps[:, :, :, :2]

        # split real and imaginary part of complex accelerometer data
        acc_real = tf.math.real(acc)

        if hparams[HP_IMAG]:
            acc_imag = tf.math.imag(acc)
            acc = tf.stack((acc_real, acc_imag), axis=4)
        else:
            acc = acc_real[:, :, :, :, tf.newaxis]

        # split real and imaginary part of complex gyrosensor data
        gyro_real = tf.math.real(gyro)

        if hparams[HP_IMAG]:
            gyro_imag = tf.math.imag(gyro)
            gyro = tf.stack((gyro_real, gyro_imag), axis=4)
        else:
            gyro = gyro_real[:, :, :, :, tf.newaxis]

        # get real part of complex gps data
        gps = tf.math.real(gps)

        acc = self.acc_conv1(acc)
        acc = self.acc_batch_norm1(acc)
        acc = self.acc_act1(acc)
        acc = self.acc_dropout1(acc) if training else acc

        acc = self.acc_conv2(acc)
        acc = self.acc_batch_norm2(acc)
        acc = self.acc_act2(acc)
        acc = self.acc_dropout2(acc) if training else acc

        acc = self.acc_conv3(acc)
        acc = self.acc_batch_norm3(acc)
        acc = self.acc_act3(acc)

        gyro = self.gyro_conv1(gyro)
        gyro = self.gyro_batch_norm1(gyro)
        gyro = self.gyro_act1(gyro)
        gyro = self.gyro_dropout1(gyro) if training else gyro

        gyro = self.gyro_conv2(gyro)
        gyro = self.gyro_batch_norm2(gyro)
        gyro = self.gyro_act2(gyro)
        gyro = self.gyro_dropout2(gyro) if training else gyro

        gyro = self.gyro_conv3(gyro)
        gyro = self.gyro_batch_norm3(gyro)
        gyro = self.gyro_act3(gyro)

        gps = gps[:, :, :, :, tf.newaxis]

        gps = self.gps_conv1(gps)
        gps = self.gps_batch_norm1(gps)
        gps = self.gps_act1(gps)
        gps = self.gps_dropout1(gps) if training else gps

        gps = self.gps_conv2(gps)
        gps = self.gps_batch_norm2(gps)
        gps = self.gps_act2(gps)
        gps = self.gps_dropout2(gps) if training else gps

        gps = self.gps_conv3(gps)
        gps = self.gps_batch_norm3(gps)
        gps = self.gps_act3(gps)

        sensor = tf.concat([acc, gyro, gps], 3)

        sensor = self.sensor_dropout(sensor)

        sensor = self.sensor_conv1(sensor)
        sensor = self.sensor_batch_norm1(sensor)
        sensor = self.sensor_act1(sensor)
        sensor = self.sensor_dropout1(sensor) if training else sensor

        sensor = self.sensor_conv2(sensor)
        sensor = self.sensor_batch_norm2(sensor)
        sensor = self.sensor_act2(sensor)
        sensor = self.sensor_dropout2(sensor) if training else sensor

        sensor = self.sensor_conv3(sensor)
        sensor = self.sensor_batch_norm3(sensor)
        sensor = self.sensor_act3(sensor)
        sensor = self.sensor_dropout3(sensor) if training else sensor

        sensor = tf.transpose(sensor, perm=(0, 2, 1, 3, 4))

        sensor = self.sensor_reshape(sensor)

        sensor = self.sensor_rnn_dropout(sensor) if training else self.sensor_rnn(sensor)

        # if hparams[HP_REDUCE_MEAN]:
        #
        #     sensor = tf.math.reduce_mean(sensor, axis=1, keepdims=False)
        #
        # else:

        sensor = self.flatten(sensor)

        sensor = self.fc(sensor)

        return sensor


def train(run_dir, hparams, train_ds, val_ds, class_weight, input_shape, tn, fp, fn, tp, auc, tss, sas, num_epochs=10,
          patience=1):
    tf.keras.backend.clear_session()
    model = DeepSense(hparams, input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR])
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tn, fp, fn, tp, auc, tss, sas])

    # Create a callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=patience,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial

        hist = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[es_callback],
                         class_weight=class_weight)

        eval_res = model.evaluate(val_ds)

        tn, fp, fn, tp = eval_res[2], eval_res[3], eval_res[4], eval_res[5]
        auc = eval_res[6]
        tss = eval_res[7]
        sas = eval_res[8]

        step = len(hist.history['loss'])

        tf.summary.scalar(METRIC_TN, tn, step=step)
        tf.summary.scalar(METRIC_FP, fp, step=step)
        tf.summary.scalar(METRIC_FN, fn, step=step)
        tf.summary.scalar(METRIC_TP, tp, step=step)
        tf.summary.scalar(METRIC_AUC, auc, step=step)
        tf.summary.scalar(METRIC_TSS, tss, step=step)
        tf.summary.scalar(METRIC_SAS, sas, step=step)


if __name__ == '__main__':
    dir = '../Ride_Data'
    checkpoint_dir = 'checkpoints/cnn/training'
    target_region = 'Berlin'
    hparam_logs = 'logs/hparam_tuning'
    class_counts_file = 'class_counts.csv'
    bucket_size = 100
    batch_size = 128
    num_epochs = 1000
    patience = 5
    in_memory = False
    deepsense = True
    fft_window = 8
    image_width = 20
    hpo_epochs = 100
    tn = tf.keras.metrics.TrueNegatives(name='tn')
    fp = tf.keras.metrics.FalsePositives(name='fp')
    fn = tf.keras.metrics.FalseNegatives(name='fn')
    tp = tf.keras.metrics.TruePositives(name='tp')
    auc = tf.keras.metrics.AUC(curve='PR', from_logits=False)
    tss = TSS()
    sas = tf.keras.metrics.SensitivityAtSpecificity(0.96, name='sas')

    if deepsense:
        input_shape = (None, fft_window, image_width, 3, 2)
    else:
        input_shape = (None, 4, int(bucket_size / 4), 8)

    train_ds, val_ds, test_ds, class_weight = load_data(dir, target_region, input_shape, batch_size, in_memory, fourier,
                                                        class_counts_file)

    HP_NUM_KERNELS_L1 = hp.HParam('num_kernels_l1', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L2 = hp.HParam('num_kernels_l2', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L3 = hp.HParam('num_kernels_l3', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L4 = hp.HParam('num_kernels_l4', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L5 = hp.HParam('num_kernels_l5', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L6 = hp.HParam('num_kernels_l6', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L7 = hp.HParam('num_kernels_l7', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L8 = hp.HParam('num_kernels_l8', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L9 = hp.HParam('num_kernels_l9', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L10 = hp.HParam('num_kernels_l10', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L11 = hp.HParam('num_kernels_l11', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L12 = hp.HParam('num_kernels_l12', hp.Discrete([8, 16, 32, 64, 128]))
    HP_DROPOUT_L1 = hp.HParam('dropout_l1', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L2 = hp.HParam('dropout_l2', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L3 = hp.HParam('dropout_l3', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L4 = hp.HParam('dropout_l4', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L5 = hp.HParam('dropout_l5', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L6 = hp.HParam('dropout_l6', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L7 = hp.HParam('dropout_l7', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L8 = hp.HParam('dropout_l8', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L9 = hp.HParam('dropout_l9', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L10 = hp.HParam('dropout_l10', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L11 = hp.HParam('dropout_l11', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L12 = hp.HParam('dropout_l12', hp.Discrete([0.25, 0.5, 0.75]))
    HP_DROPOUT_L13 = hp.HParam('dropout_l13', hp.Discrete([0.25, 0.5, 0.75]))
    HP_RNN_UNITS = hp.HParam('rnn_units', hp.Discrete([32, 64, 128, 256, 512]))
    HP_RNN_CELL_TYPE = hp.HParam('rnn_cell_type', hp.Discrete(['stacked_RNN', 'LSTM', 'None']))
    HP_REDUCE_MEAN = hp.HParam('reduce_mean', hp.Discrete([True, False]))
    HP_IMAG = hp.HParam('imag', hp.Discrete([True, False]))
    HP_LR = hp.HParam('learning_rate', hp.Discrete([0.01, 0.001, 0.0001]))

    METRIC_TN = 'val_tn'
    METRIC_FP = 'val_fp'
    METRIC_FN = 'val_fn'
    METRIC_TP = 'val_tp'
    METRIC_AUC = 'val_auc'
    METRIC_TSS = 'val_tss'
    METRIC_SAS = 'val_sas'

    with tf.summary.create_file_writer(hparam_logs).as_default():
        hp.hparams_config(
            hparams=[HP_NUM_KERNELS_L1, HP_NUM_KERNELS_L2, HP_NUM_KERNELS_L3, HP_NUM_KERNELS_L4, HP_NUM_KERNELS_L5,
                     HP_NUM_KERNELS_L6, HP_DROPOUT_L1, HP_DROPOUT_L2, HP_DROPOUT_L3, HP_DROPOUT_L4,
                     HP_DROPOUT_L5, HP_DROPOUT_L6, HP_DROPOUT_L7, HP_DROPOUT_L8, HP_DROPOUT_L9, HP_DROPOUT_L10,
                     HP_DROPOUT_L11, HP_DROPOUT_L12, HP_DROPOUT_L13, HP_RNN_UNITS, HP_RNN_CELL_TYPE, HP_IMAG, HP_LR],
            metrics=[hp.Metric(METRIC_TN, display_name='val_tn'), hp.Metric(METRIC_FP, display_name='val_fp'),
                     hp.Metric(METRIC_FN, display_name='val_fn'), hp.Metric(METRIC_TP, display_name='val_tp'),
                     hp.Metric(METRIC_AUC, display_name='val_auc'), hp.Metric(METRIC_TSS, display_name='val_tss'),
                     hp.Metric(METRIC_SAS, display_name='val_sas')],
        )

    session_num = 0

    for i in range(hpo_epochs):
        hparams = {
            HP_NUM_KERNELS_L1: HP_NUM_KERNELS_L1.domain.sample_uniform(),
            HP_NUM_KERNELS_L2: HP_NUM_KERNELS_L2.domain.sample_uniform(),
            HP_NUM_KERNELS_L3: HP_NUM_KERNELS_L3.domain.sample_uniform(),
            HP_NUM_KERNELS_L4: HP_NUM_KERNELS_L4.domain.sample_uniform(),
            HP_NUM_KERNELS_L5: HP_NUM_KERNELS_L5.domain.sample_uniform(),
            HP_NUM_KERNELS_L6: HP_NUM_KERNELS_L6.domain.sample_uniform(),
            HP_DROPOUT_L1: HP_DROPOUT_L1.domain.sample_uniform(),
            HP_DROPOUT_L2: HP_DROPOUT_L2.domain.sample_uniform(),
            HP_DROPOUT_L3: HP_DROPOUT_L3.domain.sample_uniform(),
            HP_DROPOUT_L4: HP_DROPOUT_L4.domain.sample_uniform(),
            HP_DROPOUT_L5: HP_DROPOUT_L5.domain.sample_uniform(),
            HP_DROPOUT_L6: HP_DROPOUT_L6.domain.sample_uniform(),
            HP_DROPOUT_L7: HP_DROPOUT_L7.domain.sample_uniform(),
            HP_DROPOUT_L8: HP_DROPOUT_L8.domain.sample_uniform(),
            HP_DROPOUT_L9: HP_DROPOUT_L9.domain.sample_uniform(),
            HP_DROPOUT_L10: HP_DROPOUT_L10.domain.sample_uniform(),
            HP_DROPOUT_L11: HP_DROPOUT_L11.domain.sample_uniform(),
            HP_DROPOUT_L12: HP_DROPOUT_L12.domain.sample_uniform(),
            HP_DROPOUT_L13: HP_DROPOUT_L13.domain.sample_uniform(),
            HP_RNN_UNITS: HP_RNN_UNITS.domain.sample_uniform(),
            HP_RNN_CELL_TYPE: HP_RNN_CELL_TYPE.domain.sample_uniform(),
            HP_IMAG: HP_IMAG.domain.sample_uniform(),
            HP_LR: HP_LR.domain.sample_uniform(),
        }

        print('')
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        train(hparam_logs + '_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + run_name, hparams, train_ds, val_ds,
              class_weight, input_shape, tn, fp, fn, tp, auc, tss, num_epochs, patience)
        session_num += 1
