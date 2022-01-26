import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from pyts.image import GramianAngularField

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


def data_gen(dir, split, region, deterministic=True, transpose_flag=False, gaf_flag=False):
    with np.load(os.path.join(dir, split, region)) as data:

        if deterministic:
            files = data.files
        else:
            files = np.random.choice(data.files, size=len(data.files), replace=False)

        for file in files:
            y = tf.cast(data[file][:, :, -1], tf.dtypes.int32)
            x = data[file][:, :, :-1]
            y = tf.math.reduce_mean(y, axis=0)
            y = tf.math.reduce_mean(y, axis=0)

            if transpose_flag:
                x = np.transpose(x, axes=(1, 0, 2))

                if gaf_flag:
                    x = np.reshape(x, (input_shape_global[1], input_shape_global[3]))
                    transformer = GramianAngularField(sample_range=None)

                    subs = []
                    for i in range(input_shape_global[3]):
                        # transformer expects value between -1 and 1
                        # this should generally be the case due to the scaling in the preprocessing
                        # however in some cases it values can be outside these boundaries
                        x = np.clip(x, -1, 1)
                        x_transformed = transformer.transform(x[np.newaxis, :, i])
                        subs.append(x_transformed.squeeze())

                    x = np.stack(subs, axis=2)

            yield x, y


def create_ds(dir, region, split, batch_size=32, in_memory_flag=True, count=False, deterministic=True, transpose_flag=False, gaf_flag=False,
              class_counts_file='class_counts.csv', filter_fn=None, cache_dir=None):
    if in_memory_flag:

        with np.load(os.path.join(dir, split, region + '.npz')) as data:
            y = tf.cast(data['arr_0'][:, :, :, -1], tf.dtypes.int32)
            x = data['arr_0'][:, :, :, :-1]
            y = tf.math.reduce_mean(y, axis=1)
            y = tf.math.reduce_mean(y, axis=1)
            y = y[:, tf.newaxis]

            if transpose_flag:
                x = tf.transpose(x, perm=(0, 2, 1, 3))

        ds = tf.data.Dataset.from_tensor_slices((x, y))

    else:

        ds = tf.data.Dataset.from_generator(data_gen, args=[dir, split, region + '.npz', deterministic, transpose_flag, gaf_flag],
                                            output_signature=(
                                                tf.TensorSpec(shape=(input_shape_global[1], input_shape_global[2],
                                                                     input_shape_global[3]), dtype=tf.float32),
                                                tf.TensorSpec(shape=(), dtype=tf.int32)
                                            ))

    ds = ds.filter(filter_fn) if filter_fn is not None else ds
    ds = ds.cache(cache_dir + '_' + split) if cache_dir is not None else ds
    ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    if count:
        class_counts_df = pd.read_csv(os.path.join(dir, class_counts_file))
        pos_counter, neg_counter = class_counts_df[split + '_' + region]

        return ds, pos_counter, neg_counter

    else:
        return ds


def load_data(dir, region, input_shape=(None, 5, 20, 8), batch_size=32, in_memory_flag=True, deterministic=True, transpose_flag=False,
              gaf_flag=False, class_counts_file='class_counts.csv', cache_dir=None):
    set_input_shape_global(input_shape)

    train_ds, pos_train_counter, neg_train_counter = create_ds(dir=dir, region=region, split='train', batch_size=batch_size, in_memory_flag=in_memory_flag, count=True,
                                                               deterministic=deterministic, transpose_flag=transpose_flag, gaf_flag=gaf_flag,
    class_counts_file=class_counts_file, filter_fn=None, cache_dir=cache_dir)
    val_ds = create_ds(dir=dir, region=region, split='val', batch_size=batch_size, in_memory_flag=in_memory_flag, count=False, deterministic=deterministic,
                       transpose_flag=transpose_flag, gaf_flag=gaf_flag, class_counts_file=class_counts_file, filter_fn=None, cache_dir=cache_dir)
    test_ds = create_ds(dir=dir, region=region, split='test', batch_size=batch_size, in_memory_flag=in_memory_flag, count=False, deterministic=deterministic,
                        transpose_flag=transpose_flag, gaf_flag=gaf_flag, class_counts_file=class_counts_file, filter_fn=None, cache_dir=cache_dir)

    weight_for_0 = (1 / neg_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)
    weight_for_1 = (1 / pos_train_counter) * ((pos_train_counter + neg_train_counter) / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    return train_ds, val_ds, test_ds, class_weight
