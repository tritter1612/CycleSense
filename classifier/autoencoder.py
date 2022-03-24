import os
import sys
import argparse as arg
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape

from data_loader import load_data

tf.get_logger().setLevel(logging.ERROR)


def autoencoder():
    x = tf.keras.layers.Input(shape=(20, 5, 8))

    encode = Flatten()(x)
    encode = Flatten()(encode)
    encode = Dense(32 * 6, activation="relu")(encode)
    encode = Dense(16 * 6, activation="relu")(encode)
    encode = Dense(8 * 6, activation="relu")(encode)

    decode = Dense(16 * 6, activation="relu")(encode)
    decode = Dense(32 * 6, activation="relu")(decode)
    decode = Dense(800, activation="sigmoid")(decode)
    decode = Reshape((100, 8))(decode)
    decode = Reshape((20, 5, 8))(decode)

    ae = tf.keras.Model(x, decode)

    return ae


def classifier(autoencoder_checkpoint_dir):
    x = tf.keras.layers.Input(shape=(20, 5, 8))

    encode1 = Flatten()(x)
    encode2 = Flatten()(encode1)
    encode3 = Dense(32 * 6, activation="relu", trainable=False)(encode2)
    encode4 = Dense(16 * 6, activation="relu", trainable=False)(encode3)
    encode5 = Dense(8 * 6, activation="relu", trainable=False)(encode4)

    decode1 = Dense(16 * 6, activation="relu")(encode5)
    decode2 = Dense(32 * 6, activation="relu")(decode1)
    decode3 = Dense(800, activation="sigmoid")(decode2)
    decode4 = Reshape((100, 8))(decode3)
    decode5 = Reshape((20, 5, 8))(decode4)

    autoencoder = tf.keras.Model(x, decode5)

    autoencoder.load_weights(tf.train.latest_checkpoint(
        os.path.dirname(autoencoder_checkpoint_dir)))

    pred1 = Dense(8 * 4, activation='relu')(encode5)
    pred2 = Dense(8 * 2, activation='relu')(pred1)
    pred3 = Dense(8 * 1, activation='relu')(pred2)
    pred4 = Dense(4, activation='relu')(pred3)
    pred5 = Dense(1, activation='sigmoid')(pred4)

    cl = tf.keras.Model(x, pred5)

    return cl


def train_autoencoder(train_ds, val_ds, num_epochs=10, patience=1,
                      autoencoder_checkpoint_dir='checkpoints/autoencoder_classifier/training'):
    model = autoencoder()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer, loss=tf.keras.losses.MeanAbsoluteError())

    for data in train_ds:
        train_data, train_labels = data

    for data in val_ds:
        val_data, val_labels = data

    latest = tf.train.latest_checkpoint(os.path.dirname(autoencoder_checkpoint_dir))
    try:
        model.load_weights(latest)
    except:
        print('There is no existing autoencoder checkpoint')

    # Create a callback that saves the model's weights
    cp_callback_autoencoder = tf.keras.callbacks.ModelCheckpoint(
        filepath=autoencoder_checkpoint_dir,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        save_freq='epoch')

    # Create a callback for early stopping
    es_callback_autoencoder = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=num_epochs,
              callbacks=[es_callback_autoencoder, cp_callback_autoencoder])


def train_classifier(train_ds, val_ds, test_ds, class_weight={0: 0.5, 1: 0.5}, num_epochs=10, patience=1,
                     autoencoder_checkpoint_dir='checkpoints/autoencoder/training',
                     classifier_checkpoint_dir='checkpoints/autoencoder_classifier/training'):
    model = classifier(autoencoder_checkpoint_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    metrics = ['accuracy', tf.keras.metrics.TrueNegatives(name='tn'),
               tf.keras.metrics.FalsePositives(name='fp'),
               tf.keras.metrics.FalseNegatives(name='fn'), tf.keras.metrics.TruePositives(name='tp'),
               tf.keras.metrics.AUC(curve='roc', from_logits=False, name='aucroc')
               ]

    model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=metrics)

    latest = tf.train.latest_checkpoint(os.path.dirname(classifier_checkpoint_dir))
    try:
        model.load_weights(latest)
    except:
        print('There is no existing classifier checkpoint')

    cp_callback_classifier = tf.keras.callbacks.ModelCheckpoint(
        filepath=classifier_checkpoint_dir,
        monitor='val_aucroc',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=True,
        save_freq='epoch')

    # Create a callback for early stopping
    es_callback_classifier = tf.keras.callbacks.EarlyStopping(
        monitor='val_aucroc',
        patience=patience,
        verbose=1,
        mode='max',
        restore_best_weights=True)

    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, class_weight=class_weight,
              callbacks=[es_callback_classifier, cp_callback_classifier])

    print('Model evaluation on train set:')
    model.evaluate(train_ds)
    print('Model evaluation on val set:')
    model.evaluate(val_ds)
    print('Model evaluation on test set:')
    model.evaluate(test_ds)


def main(argv):
    parser = arg.ArgumentParser(description='autoencoder')
    parser.add_argument('dir', metavar='<directory>', type=str, help='path to the data directory')
    parser.add_argument('--region', metavar='<region>', type=str, help='target region', required=False,
                        default='Berlin')
    parser.add_argument('--autoencoder_checkpoint_dir', metavar='<directory>', type=str,
                        help='checkpoint path autoencoder',
                        required=False, default='checkpoints/autoencoder/training')
    parser.add_argument('--classifier_checkpoint_dir', metavar='<directory>', type=str,
                        help='checkpoint path classifier',
                        required=False, default='checkpoints/autoencoder_classifier/training')
    parser.add_argument('--batch_size', metavar='<int>', type=int, help='batch size', required=False,
                        default=sys.maxsize)
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

    train_ds, val_ds, test_ds, class_weight = load_data(args.dir, args.region, input_shape=input_shape,
                                                        batch_size=args.batch_size,
                                                        in_memory_flag=args.in_memory_flag, transpose_flag=True,
                                                        class_counts_file=args.class_counts_file,
                                                        cache_dir=args.cache_dir)

    train_autoencoder(train_ds, val_ds, args.num_epochs, args.patience, args.autoencoder_checkpoint_dir)
    train_classifier(train_ds, val_ds, test_ds, class_weight, args.num_epochs, args.patience,
                     args.autoencoder_checkpoint_dir,
                     args.classifier_checkpoint_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
