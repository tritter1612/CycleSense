
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('data_dir','../../Data/Datasets/dataset6.csv', 'path to dataset')
flags.DEFINE_integer('epochs', 5, 'number of training epochs')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/cnn/training', 'path to save model')

class CNN(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(Dense(2000, activation='relu', input_shape=(14, 1),
                    kernel_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-4)))
        self.add(Dense(2000, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4),
                 bias_regularizer=regularizers.l2(1e-4),
                 activity_regularizer=regularizers.l2(1e-4)
                 ))
        self.add(Dense(2000, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4),
                 bias_regularizer=regularizers.l2(1e-4),
                 activity_regularizer=regularizers.l2(1e-4)
                 ))
        self.add(Dense(2000, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4),
                 bias_regularizer=regularizers.l2(1e-4),
                 activity_regularizer=regularizers.l2(1e-4)
                 ))
        self.add(Dense(1, activation='sigmoid',
                 kernel_regularizer=regularizers.l2(1e-4),
                 bias_regularizer=regularizers.l2(1e-4),
                 activity_regularizer=regularizers.l2(1e-4)
                 ))

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def train(_argv):
    data_dir = FLAGS.data_dir
    data = np.genfromtxt(data_dir, delimiter=',')
    X, y = data[:,:14], data[:,14]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_train, y_train = tf.convert_to_tensor(X_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test, y_test = tf.convert_to_tensor(X_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Add a channels dimension
    X_train = X_train[..., tf.newaxis]
    X_test = X_test[..., tf.newaxis]

    batch_size = 256

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    model = CNN()
    model.create_model()

    latest = tf.train.latest_checkpoint(os.path.dirname(FLAGS.checkpoint_dir))
    checkpoint_dir = FLAGS.checkpoint_dir
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

    model.summary()

    model.fit(train_ds, validation_data=test_ds, epochs=FLAGS.epochs, callbacks=[cp_callback, es_callback])

if __name__ == '__main__':
    try:
        app.run(train)
    except SystemExit:
        pass