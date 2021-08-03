import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, TimeDistributed, LSTM, ConvLSTM2D
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

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
        self.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(None, 4, 25, 8))))
        self.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        self.add(TimeDistributed(Dropout(0.5)))
        self.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.add(TimeDistributed(Flatten()))
        self.add(LSTM(100))
        self.add(Dropout(0.5))
        self.add(Dense(100, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


class Conv_LSTM_(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(4, 1, 25, 8)))
        self.add(Dropout(0.5))
        self.add(Flatten())
        self.add(Dense(512, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def train(train_ds, val_ds, test_ds, class_weight, num_epochs=10, patience=1, checkpoint_dir='checkpoints/cnn/training'):
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

    # Define the Keras TensorBoard callback.
    tb_logdir = 'tb_logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logdir)

    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[cp_callback, es_callback, tensorboard_callback], class_weight=class_weight)

    print()
    print('Model evaluation on train set after training:')
    model.evaluate(train_ds)

    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
    model.load_weights(latest)

    print('Model evaluation on train set:')
    model.evaluate(train_ds)
    print('Model evaluation on val set:')
    model.evaluate(val_ds)
    print('Model evaluation on test set:')
    model.evaluate(test_ds)

    y_pred = model.predict(test_ds)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    y_pred = np.round(y_pred)[:,0]

    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))
    print('F1 score:')
    print(f1_score(y_true, y_pred))
    print('Precision score:')
    print(precision_score(y_true, y_pred))
    print('Recall score:')
    print(recall_score(y_true, y_pred))
    print('Phi score:')
    print(matthews_corrcoef(y_true, y_pred))
    print('roc auc score:')
    print(roc_auc_score(y_true, y_pred))

if __name__ == '__main__':
    dir = '../Ride_Data'
    checkpoint_dir = 'checkpoints/cnn/training'
    target_region = 'Berlin'
    bucket_size = 100
    batch_size = (2 ** 10) * bucket_size
    num_epochs = 100
    patience = 10

    train_ds, val_ds, test_ds, class_weight = load_data(dir, target_region, batch_size, modeln='CNNLSTM')
    train(train_ds, val_ds, test_ds, class_weight, num_epochs, patience, checkpoint_dir)
