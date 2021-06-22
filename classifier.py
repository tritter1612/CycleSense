import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization


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
        self.add(Dense(1, activation='softmax'))

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def train(dir, checkpoint_dir, target_region=None, batch_size=22):
    data_train = tf.data.experimental.make_csv_dataset(os.path.join(dir, 'train', target_region, '*.csv'),
                                                       batch_size=batch_size, label_name='incident',
                                                       num_parallel_reads=int(batch_size / 22),
                                                       select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b',
                                                                       'c', 'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ', 'RC',
                                                                       'bike', 'childCheckBox', 'trailerCheckBox',
                                                                       'pLoc', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7',
                                                                       'i8', 'i9', 'i10', 'scary', 'incident'])
    data_test = tf.data.experimental.make_csv_dataset(os.path.join(dir, 'test', target_region, '*.csv'),
                                                      batch_size=batch_size, label_name='incident',
                                                      num_parallel_reads=int(batch_size / 22),
                                                      select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c',
                                                                      'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ', 'RC', 'bike',
                                                                      'childCheckBox', 'trailerCheckBox', 'pLoc', 'i1',
                                                                      'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9',
                                                                      'i10', 'scary', 'incident'])
    data_val = tf.data.experimental.make_csv_dataset(os.path.join(dir, 'val', target_region, '*.csv'),
                                                     batch_size=batch_size,
                                                     label_name='incident', num_parallel_reads=int(batch_size / 22),
                                                     select_columns=['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c',
                                                                     'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ', 'RC', 'bike',
                                                                     'childCheckBox', 'trailerCheckBox', 'pLoc', 'i1',
                                                                     'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9',
                                                                     'i10', 'scary', 'incident'])

    train_ds = data_train.map(pack_features_vector)
    val_ds = data_val.map(pack_features_vector)
    test_ds = data_test.map(pack_features_vector)

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

    # model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[cp_callback, es_callback])
    model.eval(test_ds)
