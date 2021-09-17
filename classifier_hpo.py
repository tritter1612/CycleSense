import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten, Conv1D, MaxPooling1D, Dropout, TimeDistributed, LSTM, GRU, \
    ConvLSTM2D, Conv3D, BatchNormalization, ReLU, Reshape, GRUCell, RNN, StackedRNNCells, add
from tensorboard.plugins.hparams import api as hp
import socket

from data_loader import load_data
from metrics import TSS


def create_buckets(dir, hparams, tmp_dir, target_region=None, bucket_size=22, in_memory=True, deepsense=False, class_counts_file='class_counts.csv'):

    image_width = bucket_size // hparams[HP_FFT_WINDOW]

    try:
        os.mkdir(tmp_dir)
        os.mkdir(os.path.join(tmp_dir, 'train'))
        os.mkdir(os.path.join(tmp_dir, 'test'))
        os.mkdir(os.path.join(tmp_dir, 'val'))
    except:
        pass

    class_counts_df = pd.DataFrame()

    for split in ['train', 'test', 'val']:

        for subdir in glob.glob(os.path.join(dir, split, '[!.]*')):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))

            ride_images_dict, ride_images_list = {}, []

            pos_counter, neg_counter = 0, 0

            if deepsense:

                for file in tqdm(file_list, desc='generate buckets for {} data'.format(split)):

                    arr = np.genfromtxt(file, delimiter=',', skip_header=True)

                    lat = arr[:, 0]
                    lat = lat[:, np.newaxis]
                    lon = arr[:, 1]
                    lon = lon[:, np.newaxis]
                    incident = arr[:, -1]
                    incident = incident[:, np.newaxis]

                    # remove lat, lon
                    arr = arr[:, 2:]

                    # remove incident
                    arr = arr[:, :-1]

                    # remove timestamp
                    arr = np.concatenate((arr[:, :3], arr[:, 4:]), axis=1)

                    n_window_splits = arr.shape[0] // hparams[HP_FFT_WINDOW]
                    window_split_range = n_window_splits * hparams[HP_FFT_WINDOW]

                    if n_window_splits > 0:

                        ride_images = np.stack(np.vsplit(arr[:window_split_range], n_window_splits), axis=1)
                        lat = np.stack(np.vsplit(lat[:window_split_range], n_window_splits), axis=1)
                        lon = np.stack(np.vsplit(lon[:window_split_range], n_window_splits), axis=1)
                        incident = np.stack(np.vsplit(incident[:window_split_range], n_window_splits), axis=1)

                        n_image_splits = n_window_splits // image_width
                        image_split_range = n_image_splits * image_width

                        if n_image_splits > 0:
                            ride_image_list = np.array_split(ride_images[:, :image_split_range, :], n_image_splits,
                                                             axis=1)
                            lat = np.array_split(lat[:, :image_split_range, :], n_image_splits, axis=1)
                            lon = np.array_split(lon[:, :image_split_range, :], n_image_splits, axis=1)
                            incident = np.array_split(incident[:, :image_split_range, :], n_image_splits, axis=1)

                            for i, ride_image in enumerate(ride_image_list):
                                # apply fourier transformation to data

                                if hparams[HP_FOURIER]:
                                    ride_image_transformed = np.fft.fft(ride_image, axis=0)
                                else:
                                    ride_image_transformed = ride_image

                                # append lat, lon & incident
                                ride_image_transformed = np.dstack(
                                    (ride_image_transformed, lat[i], lon[i], incident[i]))

                                if np.any(ride_image_transformed[:, :, 8]) > 0:
                                    ride_image_transformed[:, :, 8] = 1  # TODO: preserve incident type
                                    pos_counter += 1
                                    if in_memory:
                                        ride_images_list.append(ride_image_transformed)
                                    else:
                                        dict_name = os.path.basename(file).replace('.csv', '') + '_no' + str(i).zfill(
                                            5) + '_bucket_incident'
                                else:
                                    ride_image_transformed[:, :, 8] = 0
                                    neg_counter += 1
                                    if in_memory:
                                        ride_images_list.append(ride_image_transformed)
                                    else:
                                        dict_name = os.path.basename(file).replace('.csv', '') + '_no' + str(i).zfill(
                                            5) + '_bucket'

                                if not in_memory:
                                    ride_images_dict.update({dict_name : ride_image_transformed})

                    class_counts_df[split + '_' + region] = [pos_counter, neg_counter]

                class_counts_df.to_csv(os.path.join(tmp_dir, class_counts_file), ',', index=False)

                if in_memory:
                    np.savez(os.path.join(tmp_dir, split, region + '.npz'), ride_images_list)
                else:
                    np.savez(os.path.join(tmp_dir, split, region + '.npz'), **ride_images_dict)

            else:
                return


class DeepSense(tf.keras.Model):

    def __init__(self, hparams, input_shape=(None, 10, 10, 3, 2)):
        super(DeepSense, self).__init__()

        self.acc_conv1 = Conv3D(hparams[HP_NUM_KERNELS_L1], kernel_size=(3, 3, 3), activation=None, padding='valid')
        self.acc_batch_norm1 = BatchNormalization()
        self.acc_act1 = ReLU()
        self.acc_dropout1 = Dropout(0.5)

        self.acc_conv2 = Conv3D(hparams[HP_NUM_KERNELS_L2], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.acc_batch_norm2 = BatchNormalization()
        self.acc_act2 = ReLU()
        self.acc_dropout2 = Dropout(0.5)

        self.acc_conv3 = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.acc_batch_norm3 = BatchNormalization()
        self.acc_act3 = ReLU()

        self.acc_shortcut = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 3), activation=None, padding='valid')

        self.gyro_conv1 = Conv3D(hparams[HP_NUM_KERNELS_L1], kernel_size=(3, 3, 3), activation=None, padding='valid')
        self.gyro_batch_norm1 = BatchNormalization()
        self.gyro_act1 = ReLU()
        self.gyro_dropout1 = Dropout(0.5)

        self.gyro_conv2 = Conv3D(hparams[HP_NUM_KERNELS_L2], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gyro_batch_norm2 = BatchNormalization()
        self.gyro_act2 = ReLU()
        self.gyro_dropout2 = Dropout(0.5)

        self.gyro_conv3 = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gyro_batch_norm3 = BatchNormalization()
        self.gyro_act3 = ReLU()

        self.gyro_shortcut = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 3), activation=None, padding='valid')

        self.gps_conv1 = Conv3D(hparams[HP_NUM_KERNELS_L1], kernel_size=(3, 3, 2), activation=None, padding='valid')
        self.gps_batch_norm1 = BatchNormalization()
        self.gps_act1 = ReLU()
        self.gps_dropout1 = Dropout(0.5)

        self.gps_conv2 = Conv3D(hparams[HP_NUM_KERNELS_L2], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gps_batch_norm2 = BatchNormalization()
        self.gps_act2 = ReLU()
        self.gps_dropout2 = Dropout(0.5)

        self.gps_conv3 = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.gps_batch_norm3 = BatchNormalization()
        self.gps_act3 = ReLU()

        self.gps_shortcut = Conv3D(hparams[HP_NUM_KERNELS_L3], kernel_size=(3, 3, 2), activation=None, padding='valid')

        self.sensor_dropout = Dropout(0.5)

        self.sensor_conv1 = Conv3D(hparams[HP_NUM_KERNELS_L4], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm1 = BatchNormalization()
        self.sensor_act1 = ReLU()
        self.sensor_dropout1 = Dropout(0.5)

        self.sensor_conv2 = Conv3D(hparams[HP_NUM_KERNELS_L5], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm2 = BatchNormalization()
        self.sensor_act2 = ReLU()
        self.sensor_dropout2 = Dropout(0.5)

        self.sensor_conv3 = Conv3D(hparams[HP_NUM_KERNELS_L6], kernel_size=(3, 3, 1), activation=None, padding='same')
        self.sensor_batch_norm3 = BatchNormalization()
        self.sensor_act3 = ReLU()
        self.sensor_dropout3 = Dropout(0.5)

        self.sensor_shortcut = Conv3D(hparams[HP_NUM_KERNELS_L6], kernel_size=(3, 3, 1), activation=None, padding='same')

        self.sensor_reshape = Reshape((input_shape[2] - 2, (input_shape[1] - 2) * (2 + hparams[HP_GPS_ACTIVE]) * hparams[HP_NUM_KERNELS_L6]))

        if hparams[HP_RNN_CELL_TYPE] == 'stacked_RNN':

            self.sensor_gru1 = GRUCell(hparams[HP_RNN_UNITS], dropout=0.5, activation=None)
            self.sensor_gru2 = GRUCell(hparams[HP_RNN_UNITS], dropout=0.5, activation=None)
            self.sensor_stacked_rnn = RNN(StackedRNNCells([self.sensor_gru1, self.sensor_gru2]), return_sequences=True)

        elif hparams[HP_RNN_CELL_TYPE] == 'Single LSTM':

            self.sensor_lstm1 = LSTM(hparams[HP_RNN_UNITS], activation=None, return_sequences=True, dropout=0.5)

        elif hparams[HP_RNN_CELL_TYPE] == 'Double LSTM':

            self.sensor_lstm1 = LSTM(hparams[HP_RNN_UNITS], activation=None, return_sequences=True, dropout=0.5)
            self.sensor_lstm2 = LSTM(hparams[HP_RNN_UNITS], activation=None, return_sequences=True, dropout=0.5)

        elif hparams[HP_RNN_CELL_TYPE] == 'Single GRU':

            self.sensor_gru1 = GRU(hparams[HP_RNN_UNITS], activation=None, return_sequences=True, dropout=0.5)

        elif hparams[HP_RNN_CELL_TYPE] == 'Double GRU':

            self.sensor_gru1 = GRU(hparams[HP_RNN_UNITS], activation=None, return_sequences=True, dropout=0.5)
            self.sensor_gru2 = GRU(hparams[HP_RNN_UNITS], activation=None, return_sequences=True, dropout=0.5)

        elif hparams[HP_RNN_CELL_TYPE] == 'None':

            self.sensor_rnn = Lambda((lambda x: x))
            self.sensor_rnn_dropout = Dropout(0.5)

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

        if hparams[HP_GPS_ACTIVE]:

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

        else:

            sensor = tf.concat([acc, gyro], 3)

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

        if hparams[HP_RNN_CELL_TYPE] == 'stacked_RNN':

            sensor = self.sensor_stacked_rnn(sensor, training=training)

        elif hparams[HP_RNN_CELL_TYPE] == 'Single LSTM':

            sensor = self.sensor_lstm1(sensor, training=training)

        elif hparams[HP_RNN_CELL_TYPE] == 'Double LSTM':

            sensor = self.sensor_lstm1(sensor, training=training)
            sensor = self.sensor_lstm2(sensor, training=training)

        elif hparams[HP_RNN_CELL_TYPE] == 'Single GRU':

            sensor = self.sensor_gru1(sensor, training=training)

        elif hparams[HP_RNN_CELL_TYPE] == 'Double GRU':

            sensor = self.sensor_gru1(sensor, training=training)
            sensor = self.sensor_gru2(sensor, training=training)

        if hparams[HP_RNN_CELL_TYPE] == 'None':

            sensor = self.flatten(sensor)

        else:

            sensor = tf.math.reduce_mean(sensor, axis=1, keepdims=False)

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
        mode='max',
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
    dir = '../Ride_Data_before_buckets'
    checkpoint_dir = 'checkpoints/cnn/training'
    tmp_dir = 'tmp_dir'
    target_region = 'Berlin'
    hparam_logs = 'logs/hparam_tuning'
    class_counts_file = 'class_counts.csv'
    bucket_size = 100
    batch_size = 128
    num_epochs = 1000
    patience = 5
    in_memory = True
    deepsense = True
    hpo_epochs = 100
    tn = tf.keras.metrics.TrueNegatives(name='tn')
    fp = tf.keras.metrics.FalsePositives(name='fp')
    fn = tf.keras.metrics.FalseNegatives(name='fn')
    tp = tf.keras.metrics.TruePositives(name='tp')
    auc = tf.keras.metrics.AUC(curve='PR', from_logits=False)
    tss = TSS()
    sas = tf.keras.metrics.SensitivityAtSpecificity(0.96, name='sas')

    HP_FFT_WINDOW = hp.HParam('fft_window', hp.Discrete([5, 10]))
    HP_FOURIER = hp.HParam('fourier', hp.Discrete([True, False]))
    HP_NUM_KERNELS_L1 = hp.HParam('num_kernels_l1', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L2 = hp.HParam('num_kernels_l2', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L3 = hp.HParam('num_kernels_l3', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L4 = hp.HParam('num_kernels_l4', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L5 = hp.HParam('num_kernels_l5', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L6 = hp.HParam('num_kernels_l6', hp.Discrete([8, 16, 32, 64, 128]))
    HP_RNN_UNITS = hp.HParam('rnn_units', hp.Discrete([32, 64, 128, 256, 512]))
    HP_RNN_CELL_TYPE = hp.HParam('rnn_cell_type', hp.Discrete(['stacked_RNN', 'Single LSTM', 'Double LSTM', 'Single GRU', 'Double LSTM', 'None']))
    HP_IMAG = hp.HParam('imag', hp.Discrete([True, False]))
    HP_GPS_ACTIVE = hp.HParam('gps_active', hp.Discrete([True, False]))
    HP_LR = hp.HParam('learning_rate', hp.Discrete([0.0001]))

    METRIC_TN = 'val_tn'
    METRIC_FP = 'val_fp'
    METRIC_FN = 'val_fn'
    METRIC_TP = 'val_tp'
    METRIC_AUC = 'val_auc'
    METRIC_TSS = 'val_tss'
    METRIC_SAS = 'val_sas'

    with tf.summary.create_file_writer(hparam_logs).as_default():
        hp.hparams_config(
            hparams=[HP_FFT_WINDOW, HP_FOURIER,
                     HP_NUM_KERNELS_L1, HP_NUM_KERNELS_L2, HP_NUM_KERNELS_L3, HP_NUM_KERNELS_L4, HP_NUM_KERNELS_L5,
                     HP_NUM_KERNELS_L6, HP_RNN_UNITS, HP_RNN_CELL_TYPE, HP_IMAG, HP_GPS_ACTIVE, HP_LR],
            metrics=[hp.Metric(METRIC_TN, display_name='val_tn'), hp.Metric(METRIC_FP, display_name='val_fp'),
                     hp.Metric(METRIC_FN, display_name='val_fn'), hp.Metric(METRIC_TP, display_name='val_tp'),
                     hp.Metric(METRIC_AUC, display_name='val_auc'), hp.Metric(METRIC_TSS, display_name='val_tss'),
                     hp.Metric(METRIC_SAS, display_name='val_sas')],
        )

    session_num = 0

    for i in range(hpo_epochs):
        hparams = {
            HP_FFT_WINDOW: HP_FFT_WINDOW.domain.sample_uniform(),
            HP_FOURIER: HP_FOURIER.domain.sample_uniform(),
            HP_NUM_KERNELS_L1: HP_NUM_KERNELS_L1.domain.sample_uniform(),
            HP_NUM_KERNELS_L2: HP_NUM_KERNELS_L2.domain.sample_uniform(),
            HP_NUM_KERNELS_L3: HP_NUM_KERNELS_L3.domain.sample_uniform(),
            HP_NUM_KERNELS_L4: HP_NUM_KERNELS_L4.domain.sample_uniform(),
            HP_NUM_KERNELS_L5: HP_NUM_KERNELS_L5.domain.sample_uniform(),
            HP_NUM_KERNELS_L6: HP_NUM_KERNELS_L6.domain.sample_uniform(),
            HP_RNN_UNITS: HP_RNN_UNITS.domain.sample_uniform(),
            HP_RNN_CELL_TYPE: HP_RNN_CELL_TYPE.domain.sample_uniform(),
            HP_IMAG: HP_IMAG.domain.sample_uniform() if HP_FOURIER else False,
            HP_GPS_ACTIVE: HP_GPS_ACTIVE.domain.sample_uniform(),
            HP_LR: HP_LR.domain.sample_uniform(),
        }

        print('')
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})

        image_width = 100 // hparams[HP_FFT_WINDOW]

        if deepsense:
            input_shape = (None, hparams[HP_FFT_WINDOW], image_width, 3, 2)
        else:
            input_shape = (None, 4, int(bucket_size / 4), 8)

        create_buckets(dir, hparams, tmp_dir, target_region, bucket_size, in_memory, deepsense, class_counts_file)
        train_ds, val_ds, test_ds, class_weight = load_data(tmp_dir, target_region, input_shape, batch_size, in_memory, deepsense, os.path.join(tmp_dir, class_counts_file))

        train('_'.join([hparam_logs, datetime.now().strftime('%Y%m%d-%H%M%S'), socket.gethostname(), run_name]), hparams, train_ds, val_ds,
              class_weight, input_shape, tn, fp, fn, tp, auc, tss, sas, num_epochs, patience)

        os.remove(os.path.join(tmp_dir, 'train', target_region + '.npz'))
        os.remove(os.path.join(tmp_dir, 'val', target_region + '.npz'))
        os.remove(os.path.join(tmp_dir, 'test', target_region + '.npz'))
        os.remove(os.path.join(tmp_dir, class_counts_file))

        session_num += 1
