import os
import glob
import numpy as np
import tensorflow as tf

model = None


def pack_features_vector(features, labels):
    """Pack the features into a single array."""

    features = tf.stack(list(features.values()), axis=1)

    if model == 'LSTM':
        features = tf.reshape(features, (-1, 100, 8))
        labels = tf.reshape(labels, (-1, 100,))
        labels = tf.reduce_mean(labels, axis=1)

    if model == 'CNNLSTM':
        features = tf.reshape(features, (-1, 4, 25, 8))
        labels = tf.reshape(labels, (-1, 100,))
        labels = tf.reduce_mean(labels, axis=1)

    if model == 'ConvLSTM':
        features = tf.reshape(features, (-1, 4, 1, 25, 9))
        labels = tf.reshape(labels, (-1, 100,))
        labels = tf.reduce_mean(labels, axis=1)

    return features, labels


def load_data(dir, target_region, batch_size=44, modeln='DNN'):
    global model
    model = modeln

    pos_train_counter = float(len(glob.glob(os.path.join(dir, 'train', target_region, '*_bucket_incident.csv'))))
    neg_train_counter = float(len(glob.glob(os.path.join(dir, 'train', target_region, '*_bucket.csv'))))

    weight_for_0 = (1 / neg_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)
    weight_for_1 = (1 / pos_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    train_list = sorted(glob.glob(os.path.join(dir, 'train', target_region, '*.csv')))
    val_list = sorted(glob.glob(os.path.join(dir, 'val', target_region, '*.csv')))
    test_list = sorted(glob.glob(os.path.join(dir, 'test', target_region, '*.csv')))

    data_train = tf.data.experimental.make_csv_dataset(
        train_list,
        batch_size=batch_size, label_name='incident',
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=1,
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c', 'incident'])

    data_val = tf.data.experimental.make_csv_dataset(
        val_list,
        batch_size=batch_size, label_name='incident',
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=1,
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c', 'incident'])

    data_test = tf.data.experimental.make_csv_dataset(
        test_list,
        batch_size=batch_size, label_name='incident',
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=1,
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c', 'incident'])

    train_ds = data_train.map(pack_features_vector)
    val_ds = data_val.map(pack_features_vector)
    test_ds = data_test.map(pack_features_vector)

    return train_ds, val_ds, test_ds, class_weight
