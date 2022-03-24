import os
import sys
import argparse as arg
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout, TimeDistributed, LSTM
from sklearn.metrics import confusion_matrix

from data_loader import load_data
from metrics import TSS

tf.get_logger().setLevel(logging.ERROR)


class CNN_LSTM_(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        self.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        self.add(TimeDistributed(Dropout(0.5)))
        self.add(TimeDistributed(Flatten()))
        self.add(LSTM(100))
        self.add(Dropout(0.5))
        self.add(Dense(100, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))


def train(train_ds, val_ds, test_ds, class_weight, num_epochs=10, patience=1,
          checkpoint_dir='checkpoints/cnnlstm/training'):
    model = CNN_LSTM_()
    model.create_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.TrueNegatives(name='tn'),
                           tf.keras.metrics.FalsePositives(name='fp'),
                           tf.keras.metrics.FalseNegatives(name='fn'), tf.keras.metrics.TruePositives(name='tp'),
                           tf.keras.metrics.AUC(curve='roc', from_logits=False, name='aucroc'),
                           tf.keras.metrics.AUC(curve='PR', from_logits=False, name='aucpr'),
                           TSS(), tf.keras.metrics.SensitivityAtSpecificity(0.96, name='sas')
                           ])

    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
    try:
        model.load_weights(latest)
    except:
        print('There is no existing checkpoint')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        monitor='val_aucroc',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=True,
        save_freq='epoch')

    # Create a callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_aucroc',
        patience=patience,
        verbose=1,
        mode='max',
        restore_best_weights=True)

    # Define the Keras TensorBoard callback.
    tb_logdir = 'tb_logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=1)

    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs,
              callbacks=[cp_callback, es_callback, tensorboard_callback], class_weight=class_weight)

    print('Model evaluation on train set:')
    model.evaluate(train_ds)
    print('Model evaluation on val set:')
    model.evaluate(val_ds)
    print('Model evaluation on test set:')
    model.evaluate(test_ds)

    y_pred = model.predict(test_ds)

    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    y_pred = np.round(y_pred)[:, 0]

    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

    model.summary()


def main(argv):
    parser = arg.ArgumentParser(description='cnnlstm')
    parser.add_argument('dir', metavar='<directory>', type=str, help='path to the data directory')
    parser.add_argument('--region', metavar='<region>', type=str, help='target region', required=False,
                        default='Berlin')
    parser.add_argument('--checkpoint_dir', metavar='<directory>', type=str, help='checkpoint path model',
                        required=False, default='checkpoints/cnnlstm/training')
    parser.add_argument('--batch_size', metavar='<int>', type=int, help='batch size', required=False, default=512)
    parser.add_argument('--in_memory_flag', metavar='<bool>', type=bool,
                        help='whether the data was stored in one array or not', required=False, default=True)
    parser.add_argument('--num_epochs', metavar='<int>', type=int, help='training epochs', required=False,
                        default=100)
    parser.add_argument('--patience', metavar='<int>', type=int, help='patience value for early stopping',
                        required=False, default=10)
    parser.add_argument('--window_size', metavar='<int>', type=int, help='bucket height', required=False, default=5)
    parser.add_argument('--slices', metavar='<int>', type=int, help='bucket width', required=False, default=20)
    parser.add_argument('--class_counts_file', metavar='<file>', type=str, help='path to class counts file',
                        required=False, default='class_counts.csv')
    parser.add_argument('--cache_dir', metavar='<directory>', type=str, help='path to cache directory',
                        required=False,
                        default=None)
    args = parser.parse_args()

    input_shape = (None, args.slices, args.window_size, 8)

    train_ds, val_ds, test_ds, class_weight = load_data(args.dir, args.region, input_shape=input_shape, batch_size=args.batch_size,
                                                        in_memory_flag=args.in_memory_flag, transpose_flag=True,
                                                        class_counts_file=args.class_counts_file, cache_dir=args.cache_dir)
    train(train_ds, val_ds, test_ds, class_weight, args.num_epochs, args.patience, args.checkpoint_dir)


if __name__ == '__main__':
    main(sys.argv[1:])

