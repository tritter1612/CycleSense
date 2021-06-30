import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, TimeDistributed, LSTM, ConvLSTM2D
from tensorboard.plugins.hparams import api as hp

from data_loader import load_data


class CNN_LSTM_(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self, hparams):
        self.add(TimeDistributed(
            Conv1D(filters=hparams[HP_NUM_KERNELS_L1], kernel_size=3, activation='relu',
                   input_shape=(None, 2, 11, 9))))
        self.add(TimeDistributed(
            Conv1D(filters=hparams[HP_NUM_KERNELS_L2], kernel_size=3, activation='relu')))
        self.add(TimeDistributed(Dropout(hparams[HP_DROPOUT_L1])))
        self.add(TimeDistributed(MaxPooling1D()))
        self.add(TimeDistributed(Flatten()))
        self.add(LSTM(hparams[HP_LSTM_UNITS]))
        self.add(Dropout(hparams[HP_DROPOUT_L2]))
        self.add(Dense(hparams[HP_HIDDEN_UNITS], activation='relu'))
        self.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR])
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def train(run_dir, hparams, train_ds, val_ds, num_epochs=10, patience=1,
          checkpoint_dir='checkpoints/cnn/training'):
    model = CNN_LSTM_()
    model.create_model(hparams)

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

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial

        model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[cp_callback, es_callback])

        latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
        model.load_weights(latest)

        accuracy = model.evaluate(val_ds)[1]

        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


if __name__ == '__main__':
    dir = '../Ride_Data'
    checkpoint_dir = 'checkpoints/cnn/training'
    target_region = 'Berlin'
    batch_size = (2 ** 10) * 22
    num_epochs = 100
    patience = 5
    hpo_epochs = 100

    train_ds, val_ds, _ = load_data(dir, target_region, batch_size, modeln='CNNLSTM')


    HP_NUM_KERNELS_L1 = hp.HParam('num_kernels_l1', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L2 = hp.HParam('num_kernels_l2', hp.Discrete([8, 16, 32, 64, 128]))
    HP_DROPOUT_L1 = hp.HParam('dropout_l1', hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    HP_DROPOUT_L2 = hp.HParam('dropout_l2', hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    HP_LSTM_UNITS = hp.HParam('lstm_units', hp.Discrete([32, 64, 128, 256, 512]))
    HP_HIDDEN_UNITS = hp.HParam('hidden_units', hp.Discrete([32, 64, 128, 256, 512]))
    HP_LR = hp.HParam('learning_rate', hp.Discrete([0.1, 0.01, 0.001, 0.0001]))

    METRIC_ACCURACY = 'val_accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_KERNELS_L1, HP_NUM_KERNELS_L2,HP_DROPOUT_L1, HP_DROPOUT_L2, HP_LSTM_UNITS, HP_HIDDEN_UNITS, HP_LR],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='val_accuracy')],
        )

    session_num = 0

    for i in range(hpo_epochs):
        hparams = {
            HP_NUM_KERNELS_L1: HP_NUM_KERNELS_L1.domain.sample_uniform(),
            HP_NUM_KERNELS_L2: HP_NUM_KERNELS_L2.domain.sample_uniform(),
            HP_DROPOUT_L1: HP_DROPOUT_L1.domain.sample_uniform(),
            HP_DROPOUT_L2: HP_DROPOUT_L2.domain.sample_uniform(),
            HP_LSTM_UNITS: HP_LSTM_UNITS.domain.sample_uniform(),
            HP_HIDDEN_UNITS: HP_HIDDEN_UNITS.domain.sample_uniform(),
            HP_LR: HP_LR.domain.sample_uniform(),
        }

        try:
            os.rmdir(checkpoint_dir)
        except:
            pass

        print('')
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        train('logs/hparam_tuning/' + run_name, hparams, train_ds, val_ds, num_epochs, patience, checkpoint_dir)
        session_num += 1
