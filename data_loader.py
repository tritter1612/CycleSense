import os
import glob
import numpy as np
import tensorflow as tf


def pack_features_vector(features, labels):
    """Pack the features into a single array."""

    features = tf.stack(list(features.values()), axis=1)
    features = tf.reshape(features, (-1, 4, 25, 8))
    labels = tf.reshape(labels, (-1, 100,))
    labels = tf.reduce_mean(labels, axis=1)

    return features, labels


def create_ds(dir, target_region, split, batch_size=32, fourier=True, count=False):

    if fourier:

        with np.load(os.path.join(dir, split, target_region + '.npz')) as data:
            x = data['arr_0'][:, :, :, :]
            y = tf.cast(data['arr_0'][:, :, :, 8], tf.dtypes.int32)
            y = tf.math.reduce_mean(y, axis=1)
            y = tf.math.reduce_mean(y, axis=1)
            y = y[:, tf.newaxis]
            pos_counter = (tf.math.reduce_sum(y)).numpy()
            neg_counter = y.shape[0] - pos_counter

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.batch(batch_size)

    else:

        pos_counter = float(len(glob.glob(os.path.join(dir, split, target_region, '*_bucket_incident.csv'))))
        neg_counter = float(len(glob.glob(os.path.join(dir, split, target_region, '*_bucket.csv'))))

        file_list = sorted(glob.glob(os.path.join(dir, split, target_region, '*.csv')))

        data = tf.data.experimental.make_csv_dataset(
            file_list, label_name='incident',
            batch_size=batch_size,
            shuffle=False,
            num_epochs=1,
            select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c', 'incident'])

        ds = data.map(pack_features_vector)

    if count == True:
        return ds, pos_counter, neg_counter

    else:
        return ds


def load_data(dir, target_region, batch_size=32, fourier=True):

    train_ds, pos_train_counter, neg_train_counter = create_ds(dir, target_region, 'train', batch_size, fourier, True)
    val_ds = create_ds(dir, target_region, 'val', batch_size, fourier)
    test_ds = create_ds(dir, target_region, 'test', batch_size, fourier)

    train_ds = train_ds
    val_ds = val_ds
    test_ds = test_ds

    weight_for_0 = (1 / neg_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)
    weight_for_1 = (1 / pos_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    return train_ds, val_ds, test_ds, class_weight
