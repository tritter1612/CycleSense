import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

input_shape_global = None


def pack_features_vector(features, labels):
    """Pack the features into a single array."""

    features = tf.stack(list(features.values()), axis=1)
    features = tf.reshape(features, (-1, input_shape_global[1], input_shape_global[2], input_shape_global[3]))
    labels = tf.reshape(labels, (-1, input_shape_global[1] * input_shape_global[2],))
    labels = tf.reduce_mean(labels, axis=1)

    return features, labels


def data_gen(dir, split, target_region):
    with np.load(os.path.join(dir, split, target_region)) as data:
        for file in data.files:
            x = data[file][:, :, :]
            y = tf.cast(data[file][:, :, 8], tf.dtypes.int32)
            y = tf.math.reduce_mean(y, axis=0)
            y = tf.math.reduce_mean(y, axis=0)

            yield x, y


def create_ds(dir, target_region, split, batch_size=32, in_memory=False, deepsense=True, count=False,
              class_counts_file='class_counts.csv'):

    if deepsense:

        ds = tf.data.Dataset.from_generator(data_gen, args=[dir, split, target_region + '.npz'],
                                            output_signature=(
                                                tf.TensorSpec(shape=(input_shape_global[1], input_shape_global[2], 9), dtype=tf.complex64),
                                                tf.TensorSpec(shape=(), dtype=tf.int32)
                                            ))

        ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        ds = ds.cache().prefetch(1) if in_memory else ds.prefetch(1)

        if count:
            class_counts_df = pd.read_csv(class_counts_file)
            pos_counter, neg_counter = class_counts_df[split + '_' + target_region]

    else:

        if not in_memory and split == 'train':
            warnings.warn('if deepsense is False data is always processed in_memory')

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

    if count:
        return ds, pos_counter, neg_counter

    else:
        return ds


def load_data(dir, target_region, input_shape=(None, 4, 11, 8), batch_size=32, in_memory=False, deepsense=True,
              class_counts_file='class_counts.csv'):
    global input_shape_global
    input_shape_global = input_shape

    train_ds, pos_train_counter, neg_train_counter = create_ds(dir, target_region, 'train', batch_size, in_memory, deepsense, True, class_counts_file)
    val_ds = create_ds(dir, target_region, 'val', batch_size, in_memory, deepsense, False, class_counts_file)
    test_ds = create_ds(dir, target_region, 'test', batch_size, False, deepsense, False, class_counts_file)

    weight_for_0 = (1 / neg_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)
    weight_for_1 = (1 / pos_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    return train_ds, val_ds, test_ds, class_weight
