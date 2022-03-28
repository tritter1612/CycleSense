import os
import sys
import argparse as arg
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input

tf.get_logger().setLevel(logging.ERROR)

class FCN(tf.keras.models.Sequential):
    '''
    Definition of the fcn model.
    '''

    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(Dense(2000, activation='relu'))
        self.add(Dense(2000, activation='relu'))
        self.add(Dense(2000, activation='relu'))
        self.add(Dense(2000, activation='relu'))
        self.add(Dense(2000, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))


def normalize(dir):
    '''
    Scale and normalize data.
    @param dir: data directory
    '''

    scaler_std = StandardScaler()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    train_file = os.path.join(dir, 'train_dataset.csv')
    val_file = os.path.join(dir, 'val_dataset.csv')
    test_file = os.path.join(dir, 'test_dataset.csv')

    # fit scaler & encoder
    df = pd.read_csv(train_file)
    df.columns = ["speed", "mean_acc_x", "mean_acc_y", "mean_acc_z", "std_acc_x", "std_acc_y", "std_acc_z", "sma",
                  "mean_svm", "entropyX", "entropyY", "entropyZ", "bike_type", "phone_location", "incident_type"]

    # df.fillna(0, inplace=True)

    scaler_std.fit(df[["speed", "mean_acc_x", "mean_acc_y", "mean_acc_z", "std_acc_x", "std_acc_y", "std_acc_z",
                      "sma",
                      "mean_svm", "entropyX", "entropyY", "entropyZ"]])

    one_hot_encoder.fit(df[["bike_type", "phone_location"]])


    # transform data
    for file in [train_file, val_file, test_file]:
        df = pd.read_csv(file)
        df.columns = ["speed", "mean_acc_x", "mean_acc_y", "mean_acc_z", "std_acc_x", "std_acc_y", "std_acc_z",
                      "sma",
                      "mean_svm", "entropyX", "entropyY", "entropyZ", "bike_type", "phone_location",
                      "incident_type"]
        # df.fillna(0, inplace=True)

        df[["speed", "mean_acc_x", "mean_acc_y", "mean_acc_z", "std_acc_x", "std_acc_y", "std_acc_z", "sma",
            "mean_svm", "entropyX", "entropyY", "entropyZ"]] = scaler_std.transform(df[["speed", "mean_acc_x",
                                                                                        "mean_acc_y", "mean_acc_z",
                                                                                        "std_acc_x", "std_acc_y",
                                                                                        "std_acc_z", "sma",
                                                                                        "mean_svm", "entropyX",
                                                                                        "entropyY", "entropyZ"]])

        df_one_hot_encoded_cols = pd.DataFrame(one_hot_encoder.transform(df[['bike_type', 'phone_location']]).toarray(), columns=one_hot_encoder.get_feature_names_out())

        df = df.drop(columns=['bike_type', 'phone_location'])

        df = pd.concat([df, df_one_hot_encoded_cols], axis=1)

        df.to_csv(file, ',', index=False)


def pack_features_vector(features, labels):
    """Pack the features into a single array."""

    features = tf.stack(list(features.values()), axis=1)

    return features, labels


def load_data(dir, batch_size=128):

    train_file = os.path.join(dir, 'train_dataset.csv')
    val_file = os.path.join(dir, 'val_dataset.csv')
    test_file = os.path.join(dir, 'test_dataset.csv')

    columns = ['speed', 'mean_acc_x', 'mean_acc_y', 'mean_acc_z', 'std_acc_x', 'std_acc_y',
                        'std_acc_z', 'sma', 'mean_svm', 'entropyX', 'entropyY', 'entropyZ',
                        'incident_type']

    train_ds = tf.data.experimental.make_csv_dataset(
        train_file, label_name='incident_type',
        batch_size=batch_size,
        shuffle=False,
        num_epochs=1,
        select_columns=columns)


    val_ds = tf.data.experimental.make_csv_dataset(
        val_file, label_name='incident_type',
        batch_size=batch_size,
        shuffle=False,
        num_epochs=1,
        select_columns=columns)

    test_ds = tf.data.experimental.make_csv_dataset(
        test_file, label_name='incident_type',
        batch_size=batch_size,
        shuffle=False,
        num_epochs=1,
        select_columns=columns)

    train_ds = train_ds.map(pack_features_vector)
    val_ds = val_ds.map(pack_features_vector)
    test_ds = test_ds.map(pack_features_vector)

    return train_ds, val_ds, test_ds


def train_classifier(train_ds, val_ds, test_ds, num_epochs=10, patience=1, checkpoint_dir='checkpoints/cnn/training'):
    '''
    Training method for fcn model.
    @param train_ds: training dataset
    @param val_ds: validation dataset
    @param test_ds: test dataset
    @param num_epochs: number of training epochs
    @param patience: patience
    @param checkpoint_dir: checkpoint directory of fcn model
    '''

    model = FCN()
    model.create_model()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    metrics = ['accuracy', tf.keras.metrics.TrueNegatives(name='tn'),
               tf.keras.metrics.FalsePositives(name='fp'),
               tf.keras.metrics.FalseNegatives(name='fn'), tf.keras.metrics.TruePositives(name='tp'),
               tf.keras.metrics.AUC(curve='roc', from_logits=False, name='aucroc')
               ]

    model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=metrics)

    try:
        model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir)))
    except:
        print('There is no existing checkpoint')

    cp_callback_classifier = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
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

    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs,# class_weight=class_weight,
                    callbacks=[es_callback_classifier, cp_callback_classifier])

    print('Model evaluation on train set:')
    model.evaluate(train_ds)
    print('Model evaluation on val set:')
    model.evaluate(val_ds)
    print('Model evaluation on test set:')
    model.evaluate(test_ds)


def main(argv):
    parser = arg.ArgumentParser(description='fcn')
    parser.add_argument('dir', metavar='<directory>', type=str, help='path to the data directory')
    parser.add_argument('--checkpoint_dir', metavar='<directory>', type=str, help='checkpoint path model',
                        required=False, default='checkpoints/fcn/training')
    parser.add_argument('--batch_size', metavar='<int>', type=int, help='batch size', required=False, default=2048)
    parser.add_argument('--num_epochs', metavar='<int>', type=int, help='training epochs', required=False,
                        default=100)
    parser.add_argument('--patience', metavar='<int>', type=int, help='patience value for early stopping',
                        required=False, default=10)
    args = parser.parse_args()

    normalize(args.dir)
    train_ds, val_ds, test_ds = load_data(args.dir, args.batch_size)
    train_classifier(train_ds, val_ds, test_ds, args.num_epochs, args.patience, args.checkpoint_dir)


if __name__ == '__main__':
    main(sys.argv[1:])