import os
import glob
import numpy as np
import tensorflow as tf

model = None


def pack_features_vector(features, labels):
    """Pack the features into a single array."""

    features = tf.stack(list(features.values()), axis=1)

    return features, labels


def load_data(dir, target_region, batch_size=22, modeln='DNN'):
    global model
    model = modeln

    pos_train_counter = float(len(glob.glob(os.path.join(dir, 'train', target_region, '*_bucket_incident.csv'))) * 22)
    neg_train_counter = float(len(glob.glob(os.path.join(dir, 'train', target_region, '*_bucket.csv'))) * 22)

    pos_val_counter = float(len(glob.glob(os.path.join(dir, 'val', target_region, '*_bucket_incident.csv'))) * 22)
    neg_val_counter = float(len(glob.glob(os.path.join(dir, 'val', target_region, '*_bucket.csv'))) * 22)

    # incidents
    pos_data_train = tf.data.experimental.make_csv_dataset(
        os.path.join(dir, 'train', target_region, '*_bucket_incident.csv'),
        batch_size=batch_size, label_name='incident',
        num_parallel_reads=4,
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=np.trunc(neg_train_counter / pos_train_counter),
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b',
                        'c', 'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ', 'RC',
                        'bike', 'childCheckBox', 'trailerCheckBox',
                        'pLoc', 'incident'])

    # non-incidents
    neg_data_train = tf.data.experimental.make_csv_dataset(
        os.path.join(dir, 'train', target_region, '*_bucket.csv'),
        batch_size=batch_size, label_name='incident',
        num_parallel_reads=4,
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=1,
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b',
                        'c', 'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ',
                        'RC',
                        'bike', 'childCheckBox', 'trailerCheckBox',
                        'pLoc', 'incident'])

    # incidents
    pos_data_val = tf.data.experimental.make_csv_dataset(
        os.path.join(dir, 'val', target_region, '*_bucket_incident.csv'),
        batch_size=batch_size, label_name='incident',
        num_parallel_reads=4,
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=np.trunc(neg_val_counter / pos_val_counter),
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b',
                        'c', 'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ', 'RC',
                        'bike', 'childCheckBox', 'trailerCheckBox',
                        'pLoc', 'incident'])

    # non-incidents
    neg_data_val = tf.data.experimental.make_csv_dataset(
        os.path.join(dir, 'val', target_region, '*_bucket.csv'),
        batch_size=batch_size, label_name='incident',
        num_parallel_reads=4,
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=1,
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b',
                        'c', 'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ',
                        'RC',
                        'bike', 'childCheckBox', 'trailerCheckBox',
                        'pLoc', 'incident'])

    data_test = tf.data.experimental.make_csv_dataset(
        os.path.join(dir, 'test', target_region, '*.csv'),
        batch_size=batch_size, label_name='incident',
        num_parallel_reads=4,
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=1,
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c',
                        'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ', 'RC', 'bike',
                        'childCheckBox', 'trailerCheckBox', 'pLoc', 'incident'])

    pos_train_ds = pos_data_train.map(pack_features_vector)
    neg_train_ds = neg_data_train.map(pack_features_vector)
    pos_val_ds = pos_data_val.map(pack_features_vector)
    neg_val_ds = neg_data_val.map(pack_features_vector)
    test_ds = data_test.map(pack_features_vector)

    train_ds_resampled = tf.data.experimental.sample_from_datasets([pos_train_ds, neg_train_ds], weights=[0.5, 0.5])
    val_ds_resampled = tf.data.experimental.sample_from_datasets([pos_val_ds, neg_val_ds], weights=[0.5, 0.5])

    return train_ds_resampled, val_ds_resampled, test_ds
