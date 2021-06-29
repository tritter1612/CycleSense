import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, TimeDistributed, LSTM, ConvLSTM2D

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


class LSTM_(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(LSTM(100))
        self.add(Dropout(0.5))
        self.add(Dense(100, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


class CNN_LSTM_(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(None, 2, 11, 20))))
        self.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        self.add(TimeDistributed(Dropout(0.5)))
        self.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.add(TimeDistributed(Flatten()))
        self.add(LSTM(100))
        self.add(Dropout(0.5))
        self.add(Dense(100, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


class Conv_LSTM_(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(2, 1, 11, 20)))
        self.add(Dropout(0.5))
        self.add(Flatten())
        self.add(Dense(100, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def train(train_ds, val_ds, test_ds, num_epochs=10, patience=1, checkpoint_dir='checkpoints/cnn/training'):
    model = CNN_LSTM_()
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


if __name__ == '__main__':
    dir = './Ride_Data'
    checkpoint_dir = 'checkpoints/cnn/training'
    target_region = 'Berlin'
    batch_size = (2 ** 12) * 22
    num_epochs = 100
    patience = 5

    train_ds, val_ds, test_ds = load_data(dir, target_region, batch_size, modeln='CNNLSTM')
    train(train_ds, val_ds, test_ds, num_epochs, patience, checkpoint_dir)
