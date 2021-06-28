import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization, TimeDistributed, \
    LSTM

from data_loader import load_data


class DNN_(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(Dense(100, activation='relu'))
        self.add(Dense(100, activation='relu'))
        self.add(Dense(100, activation='relu'))
        self.add(Flatten())
        self.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def train(train_ds, val_ds, test_ds, num_epochs=10, patience=1, checkpoint_dir='checkpoints/cnn/training'):
    model = DNN_()
    model.create_model()

    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
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
        patience=patience,
        verbose=1)

    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[cp_callback, es_callback])

    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
    model.load_weights(latest)

    print('')
    print('Model evaluation on train set:')
    model.evaluate(train_ds)
    print('Model evaluation on val set:')
    model.evaluate(val_ds)
    print('Model evaluation on test set:')
    model.evaluate(test_ds)
