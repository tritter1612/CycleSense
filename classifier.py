import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization, TimeDistributed, \
    LSTM


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


class DNN(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(Dense(100, activation='relu'))
        self.add(Dense(100, activation='relu'))
        self.add(Dense(100, activation='relu'))
        self.add(Flatten())
        self.add(Dense(1, activation='sigmoid'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def train(dir, checkpoint_dir, target_region=None, batch_size=22):
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
                        'pLoc', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7',
                        'i8', 'i9', 'i10', 'scary', 'incident'])

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
                        'pLoc', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6',
                        'i7',
                        'i8', 'i9', 'i10', 'scary', 'incident'])

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
                        'pLoc', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7',
                        'i8', 'i9', 'i10', 'scary', 'incident'])

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
                        'pLoc', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6',
                        'i7',
                        'i8', 'i9', 'i10', 'scary', 'incident'])

    data_test = tf.data.experimental.make_csv_dataset(
        os.path.join(dir, 'test', target_region, '*.csv'),
        batch_size=batch_size, label_name='incident',
        num_parallel_reads=4,
        shuffle=False,
        prefetch_buffer_size=batch_size,
        num_epochs=1,
        select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c',
                        'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ', 'RC', 'bike',
                        'childCheckBox', 'trailerCheckBox', 'pLoc', 'i1',
                        'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9',
                        'i10', 'scary', 'incident'])

    pos_train_ds = pos_data_train.map(pack_features_vector)
    neg_train_ds = neg_data_train.map(pack_features_vector)
    pos_val_ds = pos_data_val.map(pack_features_vector)
    neg_val_ds = neg_data_val.map(pack_features_vector)
    test_ds = data_test.map(pack_features_vector)

    train_ds_resampled = tf.data.experimental.sample_from_datasets([pos_train_ds, neg_train_ds], weights=[0.5, 0.5])
    val_ds_resampled = tf.data.experimental.sample_from_datasets([pos_val_ds, neg_val_ds], weights=[0.5, 0.5])

    model = DNN()
    model.create_model()

    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
    checkpoint_dir = checkpoint_dir
    try:
        model.load_weights(latest)
    except:
        print('There is no existing checkpoint')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=True,
        save_freq='epoch')

    # Create a callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1)

    model.fit(train_ds_resampled, validation_data=val_ds_resampled, epochs=250, callbacks=[cp_callback, es_callback])
    model.evaluate(test_ds)
