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
            Conv1D(filters=hparams[HP_NUM_KERNELS_L1], kernel_size=hparams[HP_KERNEL_SIZE_L1], activation='relu',
                   input_shape=(None, 2, 11, 9))))
        self.add(TimeDistributed(
            Conv1D(filters=hparams[HP_NUM_KERNELS_L2], kernel_size=hparams[HP_KERNEL_SIZE_L2], activation='relu')))
        self.add(TimeDistributed(Dropout(hparams[HP_DROPOUT_L1])))
        self.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.add(TimeDistributed(Flatten()))
        self.add(LSTM(100))
        self.add(Dropout(hparams[HP_DROPOUT_L2]))
        self.add(Dense(100, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR])
        self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def train(run_dir, hparams, train_ds, val_ds, test_ds, num_epochs=10, patience=1,
          checkpoint_dir='checkpoints/cnn/training'):
    model = CNN_LSTM_()
    model.create_model(hparams)

    # latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))

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
    batch_size = (2 ** 12) * 22
    num_epochs = 100
    patience = 5

    HP_NUM_KERNELS_L1 = hp.HParam('num_kernels_l1', hp.Discrete([8, 16, 32, 64, 128]))
    HP_NUM_KERNELS_L2 = hp.HParam('num_kernels_l2', hp.Discrete([8, 16, 32, 64, 128]))
    HP_KERNEL_SIZE_L1 = hp.HParam('kernel_size_l1', hp.Discrete([3, 5, 7]))
    HP_KERNEL_SIZE_L2 = hp.HParam('kernel_size_l2', hp.Discrete([3, 5, 7]))
    HP_DROPOUT_L1 = hp.HParam('dropout_l1', hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    HP_DROPOUT_L2 = hp.HParam('dropout_l2', hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    HP_LR = hp.HParam('learning_rate', hp.Discrete([0.1, 0.01, 0.001, 0.0001]))

    METRIC_ACCURACY = 'val_accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_KERNELS_L1, HP_NUM_KERNELS_L2, HP_KERNEL_SIZE_L1, HP_KERNEL_SIZE_L2, HP_DROPOUT_L1,
                     HP_DROPOUT_L2, HP_LR],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='VAL_Accuracy')],
        )

    session_num = 0

    train_ds, val_ds, test_ds = load_data(dir, target_region, batch_size, modeln='CNNLSTM')

    for num_kernels_l1 in HP_NUM_KERNELS_L1.domain.values:
        for num_kernels_l2 in HP_NUM_KERNELS_L2.domain.values:
            for kernel_size_l1 in HP_KERNEL_SIZE_L1.domain.values:
                for kernel_size_l2 in HP_KERNEL_SIZE_L2.domain.values:
                    for dropout_rate_l1 in HP_DROPOUT_L1.domain.values:
                        for dropout_rate_l2 in HP_DROPOUT_L2.domain.values:
                            for learning_rate in HP_LR.domain.values:
                                hparams = {
                                    HP_NUM_KERNELS_L1: num_kernels_l1,
                                    HP_NUM_KERNELS_L2: num_kernels_l2,
                                    HP_KERNEL_SIZE_L1: kernel_size_l1,
                                    HP_KERNEL_SIZE_L2: kernel_size_l2,
                                    HP_DROPOUT_L1: dropout_rate_l1,
                                    HP_DROPOUT_L2: dropout_rate_l2,
                                    HP_LR: learning_rate,
                                }

                                try:
                                    os.rmdir(checkpoint_dir)
                                except:
                                    pass

                                print('')
                                run_name = "run-%d" % session_num
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})
                                train('logs/hparam_tuning/' + run_name, hparams, train_ds, val_ds, test_ds, num_epochs,
                                      patience,
                                      checkpoint_dir)
                                session_num += 1
