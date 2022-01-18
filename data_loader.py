import os
import numpy as np
import pandas as pd
import tensorflow as tf

input_shape_global = None


def set_input_shape_global(input_shape):
    global input_shape_global
    input_shape_global = input_shape


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
            y = tf.cast(data[file][:, :, -1], tf.dtypes.int32)
            x = data[file][:, :, :-1]
            y = tf.math.reduce_mean(y, axis=0)
            y = tf.math.reduce_mean(y, axis=0)

            yield x, y


def create_ds(dir, target_region, split, batch_size=32, in_memory_flag=True, count=False, class_counts_file='class_counts.csv', filter_fn=None):

    if in_memory_flag:

        with np.load(os.path.join(dir, split, target_region + '.npz')) as data:
            y = tf.cast(data['arr_0'][:, :, :, -1], tf.dtypes.int32)
            x = data['arr_0'][:, :, :, :-1]
            y = tf.math.reduce_mean(y, axis=1)
            y = tf.math.reduce_mean(y, axis=1)
            y = y[:, tf.newaxis]

        ds = tf.data.Dataset.from_tensor_slices((x, y))

    else:

        ds = tf.data.Dataset.from_generator(data_gen, args=[dir, split, target_region + '.npz'],
                                            output_signature=(
                                                tf.TensorSpec(
                                                    shape=(
                                                        input_shape_global[1], input_shape_global[2], 8),
                                                    dtype=tf.complex64),
                                                tf.TensorSpec(shape=(), dtype=tf.int32)
                                            ))

    ds = ds.filter(filter_fn) if filter_fn is not None else ds
    ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    if count:
        class_counts_df = pd.read_csv(os.path.join(dir, class_counts_file))
        pos_counter, neg_counter = class_counts_df[split + '_' + target_region]

    if count:
        return ds, pos_counter, neg_counter

    else:
        return ds


def load_data(dir, target_region, input_shape=(None, 5, 20, 3, 2), batch_size=32, in_memory_flag=True,
              class_counts_file='class_counts.csv'):
    set_input_shape_global(input_shape)

    train_ds, pos_train_counter, neg_train_counter = create_ds(dir, target_region, 'train', batch_size, in_memory_flag, True, class_counts_file)
    val_ds = create_ds(dir, target_region, 'val', batch_size, in_memory_flag, False, class_counts_file)
    test_ds = create_ds(dir, target_region, 'test', batch_size, in_memory_flag, False, class_counts_file)

    weight_for_0 = (1 / neg_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)
    weight_for_1 = (1 / pos_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    return train_ds, val_ds, test_ds, class_weight
