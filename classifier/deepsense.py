import os
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, RNN, GRUCell, StackedRNNCells, Reshape, BatchNormalization, \
    ReLU, Dropout, add, Input
tf.get_logger().setLevel(logging.ERROR)
from sklearn.metrics import confusion_matrix

from data_loader import load_data
from metrics import TSS


def submodel_acc(input_shape=(None, 5, 20, 14), output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    acc = x[:, :, :, :3]

    acc = acc[:, :, :, :, tf.newaxis]

    acc_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape)(acc)
    acc_conv1 = BatchNormalization()(acc_conv1)
    acc_conv1 = ReLU()(acc_conv1)
    acc_conv1 = Dropout(0.5)(acc_conv1)

    acc_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(acc_conv1)
    acc_conv2 = BatchNormalization()(acc_conv2)
    acc_conv2 = ReLU()(acc_conv2)
    acc_conv2 = Dropout(0.5)(acc_conv2)

    acc_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(acc_conv2)
    acc_conv3 = BatchNormalization()(acc_conv3)
    acc_conv3 = ReLU()(acc_conv3)

    acc_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid')(acc)
    acc_shortcut = BatchNormalization()(acc_shortcut)
    acc_shortcut = ReLU()(acc_shortcut)

    acc = add([acc_conv3, acc_shortcut])

    acc_fc = Flatten()(acc)
    acc_fc = Dense(1, activation='sigmoid', bias_initializer=output_bias)(acc_fc)

    model = Model(x, acc_fc)

    return model


def submodel_acc_imag(input_shape=(None, 5, 20, 14), output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    acc = x[:, :, :, 6:9]

    acc = acc[:, :, :, :, tf.newaxis]

    acc_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape)(acc)
    acc_conv1 = BatchNormalization()(acc_conv1)
    acc_conv1 = ReLU()(acc_conv1)
    acc_conv1 = Dropout(0.5)(acc_conv1)

    acc_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(acc_conv1)
    acc_conv2 = BatchNormalization()(acc_conv2)
    acc_conv2 = ReLU()(acc_conv2)
    acc_conv2 = Dropout(0.5)(acc_conv2)

    acc_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(acc_conv2)
    acc_conv3 = BatchNormalization()(acc_conv3)
    acc_conv3 = ReLU()(acc_conv3)

    acc_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid')(acc)
    acc_shortcut = BatchNormalization()(acc_shortcut)
    acc_shortcut = ReLU()(acc_shortcut)

    acc = add([acc_conv3, acc_shortcut])

    acc_fc = Flatten()(acc)
    acc_fc = Dense(1, activation='sigmoid', bias_initializer=output_bias)(acc_fc)

    model = Model(x, acc_fc)

    return model


def submodel_gyro(input_shape=(None, 5, 20, 14), output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    gyro = x[:, :, :, 3:6]

    gyro = gyro[:, :, :, :, tf.newaxis]

    gyro_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape)(gyro)
    gyro_conv1 = BatchNormalization()(gyro_conv1)
    gyro_conv1 = ReLU()(gyro_conv1)
    gyro_conv1 = Dropout(0.5)(gyro_conv1)

    gyro_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(gyro_conv1)
    gyro_conv2 = BatchNormalization()(gyro_conv2)
    gyro_conv2 = ReLU()(gyro_conv2)
    gyro_conv2 = Dropout(0.5)(gyro_conv2)

    gyro_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(gyro_conv2)
    gyro_conv3 = BatchNormalization()(gyro_conv3)
    gyro_conv3 = ReLU()(gyro_conv3)

    gyro_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid')(gyro)
    gyro_shortcut = BatchNormalization()(gyro_shortcut)
    gyro_shortcut = ReLU()(gyro_shortcut)

    gyro = add([gyro_conv3, gyro_shortcut])

    gyro_fc = Flatten()(gyro)
    gyro_fc = Dense(1, activation='sigmoid', bias_initializer=output_bias)(gyro_fc)

    model = Model(x, gyro_fc)

    return model


def submodel_gyro_imag(input_shape=(None, 5, 20, 14), output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    gyro = x[:, :, :, 9:12]

    gyro = gyro[:, :, :, :, tf.newaxis]

    gyro_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape)(gyro)
    gyro_conv1 = BatchNormalization()(gyro_conv1)
    gyro_conv1 = ReLU()(gyro_conv1)
    gyro_conv1 = Dropout(0.5)(gyro_conv1)

    gyro_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(gyro_conv1)
    gyro_conv2 = BatchNormalization()(gyro_conv2)
    gyro_conv2 = ReLU()(gyro_conv2)
    gyro_conv2 = Dropout(0.5)(gyro_conv2)

    gyro_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(gyro_conv2)
    gyro_conv3 = BatchNormalization()(gyro_conv3)
    gyro_conv3 = ReLU()(gyro_conv3)

    gyro_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid')(gyro)
    gyro_shortcut = BatchNormalization()(gyro_shortcut)
    gyro_shortcut = ReLU()(gyro_shortcut)

    gyro = add([gyro_conv3, gyro_shortcut])

    gyro_fc = Flatten()(gyro)
    gyro_fc = Dense(1, activation='sigmoid', bias_initializer=output_bias)(gyro_fc)

    model = Model(x, gyro_fc)

    return model


def submodel_gps(input_shape=(None, 5, 20, 14), output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    gps = x[:, :, :, 12:]

    gps = tf.math.real(gps)[:, :, :, :, tf.newaxis]

    gps_conv1 = Conv3D(64, kernel_size=(3, 3, 2), padding='valid')(gps)
    gps_conv1 = BatchNormalization()(gps_conv1)
    gps_conv1 = ReLU()(gps_conv1)
    gps_conv1 = Dropout(0.5)(gps_conv1)

    gps_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(gps_conv1)
    gps_conv2 = BatchNormalization()(gps_conv2)
    gps_conv2 = ReLU()(gps_conv2)
    gps_conv2 = Dropout(0.5)(gps_conv2)

    gps_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(gps_conv2)
    gps_conv3 = BatchNormalization()(gps_conv3)
    gps_conv3 = ReLU()(gps_conv3)

    gps_shortcut = Conv3D(64, kernel_size=(3, 3, 2), padding='valid')(gps)
    gps_shortcut = BatchNormalization()(gps_shortcut)
    gps_shortcut = ReLU()(gps_shortcut)

    gps = add([gps_conv3, gps_shortcut])

    gps_fc = Flatten()(gps)
    gps_fc = Dense(1, activation='sigmoid', bias_initializer=output_bias)(gps_fc)

    model = Model(x, gps_fc)

    return model


def deepsense_model(input_shape=(None, 5, 20, 14), output_bias=None, stacking=True, freeze=True,
                    ckpt_deepsense_acc='checkpoints/deepsense_sub_acc/training',
                    ckpt_deepsense_acc_imag='checkpoints/deepsense_sub_acc_imag/training',
                    ckpt_deepsense_gyro='checkpoints/deepsense_sub_gyro/training',
                    ckpt_deepsense_gyro_imag='checkpoints/deepsense_sub_gyro_imag/training',
                    ckpt_deepsense_gps='checkpoints/deepsense_sub_gps/training'):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    acc_real = x[:, :, :, 0:3]
    gyro_real = x[:, :, :, 3:6]
    acc_imag = x[:, :, :, 6:9]
    gyro_imag = x[:, :, :, 9:12]
    gps = x[:, :, :, 12:]

    acc_real = acc_real[:, :, :, :, tf.newaxis]
    gyro_real = gyro_real[:, :, :, :, tf.newaxis]

    acc_imag = acc_imag[:, :, :, :, tf.newaxis]
    gyro_imag = gyro_imag[:, :, :, :, tf.newaxis]

    gps = tf.math.real(gps)[:, :, :, :, tf.newaxis]

    acc_real_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape, trainable=not freeze)(acc_real)
    acc_real_conv1 = BatchNormalization()(acc_real_conv1)
    acc_real_conv1 = ReLU()(acc_real_conv1)
    acc_real_conv1 = Dropout(0.5)(acc_real_conv1)

    acc_real_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(acc_real_conv1)
    acc_real_conv2 = BatchNormalization()(acc_real_conv2)
    acc_real_conv2 = ReLU()(acc_real_conv2)
    acc_real_conv2 = Dropout(0.5)(acc_real_conv2)

    acc_real_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(acc_real_conv2)
    acc_real_conv3 = BatchNormalization()(acc_real_conv3)
    acc_real_conv3 = ReLU()(acc_real_conv3)

    acc_real_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', trainable=not freeze)(acc_real)
    acc_real_shortcut = BatchNormalization()(acc_real_shortcut)
    acc_real_shortcut = ReLU()(acc_real_shortcut)

    acc_real = add([acc_real_conv3, acc_real_shortcut])

    gyro_real_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape, trainable=not freeze)(gyro_real)
    gyro_real_conv1 = BatchNormalization()(gyro_real_conv1)
    gyro_real_conv1 = ReLU()(gyro_real_conv1)
    gyro_real_conv1 = Dropout(0.5)(gyro_real_conv1)

    gyro_real_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(gyro_real_conv1)
    gyro_real_conv2 = BatchNormalization()(gyro_real_conv2)
    gyro_real_conv2 = ReLU()(gyro_real_conv2)
    gyro_real_conv2 = Dropout(0.5)(gyro_real_conv2)

    gyro_real_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(gyro_real_conv2)
    gyro_real_conv3 = BatchNormalization()(gyro_real_conv3)
    gyro_real_conv3 = ReLU()(gyro_real_conv3)

    gyro_real_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', trainable=not freeze)(gyro_real)
    gyro_real_shortcut = BatchNormalization()(gyro_real_shortcut)
    gyro_real_shortcut = ReLU()(gyro_real_shortcut)

    gyro_real = add([gyro_real_conv3, gyro_real_shortcut])

    acc_imag_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape, trainable=not freeze)(acc_imag)
    acc_imag_conv1 = BatchNormalization()(acc_imag_conv1)
    acc_imag_conv1 = ReLU()(acc_imag_conv1)
    acc_imag_conv1 = Dropout(0.5)(acc_imag_conv1)

    acc_imag_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(acc_imag_conv1)
    acc_imag_conv2 = BatchNormalization()(acc_imag_conv2)
    acc_imag_conv2 = ReLU()(acc_imag_conv2)
    acc_imag_conv2 = Dropout(0.5)(acc_imag_conv2)

    acc_imag_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(acc_imag_conv2)
    acc_imag_conv3 = BatchNormalization()(acc_imag_conv3)
    acc_imag_conv3 = ReLU()(acc_imag_conv3)

    acc_imag_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', trainable=not freeze)(acc_imag)
    acc_imag_shortcut = BatchNormalization()(acc_imag_shortcut)
    acc_imag_shortcut = ReLU()(acc_imag_shortcut)

    acc_imag = add([acc_imag_conv3, acc_imag_shortcut])

    gyro_imag_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape, trainable=not freeze)(gyro_imag)
    gyro_imag_conv1 = BatchNormalization()(gyro_imag_conv1)
    gyro_imag_conv1 = ReLU()(gyro_imag_conv1)
    gyro_imag_conv1 = Dropout(0.5)(gyro_imag_conv1)

    gyro_imag_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(gyro_imag_conv1)
    gyro_imag_conv2 = BatchNormalization()(gyro_imag_conv2)
    gyro_imag_conv2 = ReLU()(gyro_imag_conv2)
    gyro_imag_conv2 = Dropout(0.5)(gyro_imag_conv2)

    gyro_imag_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(gyro_imag_conv2)
    gyro_imag_conv3 = BatchNormalization()(gyro_imag_conv3)
    gyro_imag_conv3 = ReLU()(gyro_imag_conv3)

    gyro_imag_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', trainable=not freeze)(gyro_imag)
    gyro_imag_shortcut = BatchNormalization()(gyro_imag_shortcut)
    gyro_imag_shortcut = ReLU()(gyro_imag_shortcut)

    gyro_imag = add([gyro_imag_conv3, gyro_imag_shortcut])

    gps_conv1 = Conv3D(64, kernel_size=(3, 3, 2), padding='valid', trainable=not freeze)(gps)
    gps_conv1 = BatchNormalization()(gps_conv1)
    gps_conv1 = ReLU()(gps_conv1)
    gps_conv1 = Dropout(0.5)(gps_conv1)

    gps_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(gps_conv1)
    gps_conv2 = BatchNormalization()(gps_conv2)
    gps_conv2 = ReLU()(gps_conv2)
    gps_conv2 = Dropout(0.5)(gps_conv2)

    gps_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(gps_conv2)
    gps_conv3 = BatchNormalization()(gps_conv3)
    gps_conv3 = ReLU()(gps_conv3)

    gps_shortcut = Conv3D(64, kernel_size=(3, 3, 2), padding='valid', trainable=not freeze)(gps)
    gps_shortcut = BatchNormalization()(gps_shortcut)
    gps_shortcut = ReLU()(gps_shortcut)

    gps = add([gps_conv3, gps_shortcut])

    if stacking:
        acc_fc = Flatten()(acc_real)
        acc_fc = Dense(1, activation='sigmoid')(acc_fc)

        model_acc = Model(x, acc_fc)

        gyro_fc = Flatten()(gyro_real)
        gyro_fc = Dense(1, activation='sigmoid')(gyro_fc)

        model_gyro = Model(x, gyro_fc)

        model_acc.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_deepsense_acc)))

        model_gyro.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_deepsense_gyro)))

        acc_imag_fc = Flatten()(acc_imag)
        acc_imag_fc = Dense(1, activation='sigmoid')(acc_imag_fc)

        model_acc_imag = Model(x, acc_imag_fc)

        gyro_imag_fc = Flatten()(gyro_imag)
        gyro_imag_fc = Dense(1, activation='sigmoid')(gyro_imag_fc)

        model_gyro_imag = Model(x, gyro_imag_fc)

        model_acc_imag.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_deepsense_acc_imag)))

        model_gyro_imag.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_deepsense_gyro_imag)))

        gps_fc = Flatten()(gps)
        gps_fc = Dense(1, activation='sigmoid')(gps_fc)

        model_gps = Model(x, gps_fc)

        model_gps.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_deepsense_gps)))

    sensor = tf.concat([acc_real, acc_imag, gyro_real, gyro_imag, gps], 3)

    sensor = Dropout(0.5)(sensor)

    sensor_conv1 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(sensor)
    sensor_conv1 = BatchNormalization()(sensor_conv1)
    sensor_conv1 = ReLU()(sensor_conv1)
    sensor_conv1 = Dropout(0.5)(sensor_conv1)

    sensor_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(sensor_conv1)
    sensor_conv2 = BatchNormalization()(sensor_conv2)
    sensor_conv2 = ReLU()(sensor_conv2)
    sensor_conv2 = Dropout(0.5)(sensor_conv2)

    sensor_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(sensor_conv2)
    sensor_conv3 = BatchNormalization()(sensor_conv3)
    sensor_conv3 = ReLU()(sensor_conv3)
    sensor_conv3 = Dropout(0.5)(sensor_conv3)

    sensor_conv4 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(sensor_conv3)
    sensor_conv4 = BatchNormalization()(sensor_conv4)
    sensor_conv4 = ReLU()(sensor_conv4)

    sensor_conv5 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(sensor_conv4)
    sensor_conv5 = BatchNormalization()(sensor_conv5)
    sensor_conv5 = ReLU()(sensor_conv5)

    sensor_conv6 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(sensor_conv5)
    sensor_conv6 = BatchNormalization()(sensor_conv6)
    sensor_conv6 = ReLU()(sensor_conv6)

    sensor_shortcut = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(sensor)
    sensor_shortcut = BatchNormalization()(sensor_shortcut)
    sensor_shortcut = ReLU()(sensor_shortcut)
    sensor_shortcut = Dropout(0.5)(sensor_shortcut)

    sensor = add([sensor_conv6, sensor_shortcut])

    sensor = tf.transpose(sensor, perm=(0, 2, 1, 3, 4))

    sensor = Reshape((input_shape[2] - 2, (input_shape[1] - 2) * 5 * 64))(sensor)

    sensor_gru1 = GRUCell(120, dropout=0.5, activation=None)
    sensor_gru2 = GRUCell(120, dropout=0.5, activation=None)
    sensor = RNN(StackedRNNCells([sensor_gru1, sensor_gru2]), return_sequences=True)(sensor)

    sensor = tf.math.reduce_mean(sensor, axis=1, keepdims=False)

    sensor = Dense(1, activation='sigmoid', bias_initializer=output_bias)(sensor)

    model = Model(x, sensor)

    return model


def train_submodels(train_ds, val_ds, test_ds, class_weight, num_epochs=10, patience=1, input_shape=(None, 5, 20, 14),
                    ckpt_deepsense_acc='checkpoints/deepsense_sub_acc/training',
                    ckpt_deepsense_acc_imag='checkpoints/deepsense_sub_acc_imag/training',
                    ckpt_deepsense_gyro='checkpoints/deepsense_sub_gyro/training',
                    ckpt_deepsense_gyro_imag='checkpoints/deepsense_sub_gyro_imag/training',
                    ckpt_deepsense_gps='checkpoints/deepsense_sub_gps/training'
                    ):
    initial_bias = np.log(class_weight[0] / class_weight[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),

    # Create a callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_aucroc',
        patience=patience,
        verbose=1,
        mode='max',
        restore_best_weights=True)

    for model, ckpt_deepsense_sub in [(submodel_acc(input_shape, initial_bias), ckpt_deepsense_acc),
                                      (submodel_acc_imag(input_shape, initial_bias), ckpt_deepsense_acc_imag),
                                      (submodel_gyro(input_shape, initial_bias), ckpt_deepsense_gyro),
                                      (submodel_gyro_imag(input_shape, initial_bias), ckpt_deepsense_gyro_imag),
                                      (submodel_gps(input_shape, initial_bias), ckpt_deepsense_gps)]:

        metrics = ['accuracy',
                   tf.keras.metrics.TrueNegatives(name='tn'),
                   tf.keras.metrics.FalsePositives(name='fp'),
                   tf.keras.metrics.FalseNegatives(name='fn'),
                   tf.keras.metrics.TruePositives(name='tp'),
                   tf.keras.metrics.AUC(curve='roc', from_logits=False, name='aucroc')]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        try:
            model.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_deepsense_sub)))
        except:
            print('There is no existing checkpoint')

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_deepsense_sub,
            monitor='val_aucroc',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_weights_only=True,
            save_freq='epoch')

        # Define the Keras TensorBoard callback.
        tb_logdir = 'tb_logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=1)

        model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[cp_callback, es_callback, tensorboard_callback],
                  class_weight=class_weight)


def train_deepsense(train_ds, val_ds, test_ds, class_weight, num_epochs=10, patience=1, input_shape=(None, 5, 20, 14),
                    stacking=False, freeze=False, ckpt_deepsense='checkpoints/deepsense/training',
                    ckpt_deepsense_acc='checkpoints/deepsense_sub_acc/training',
                    ckpt_deepsense_acc_imag='checkpoints/deepsense_sub_acc_imag/training',
                    ckpt_deepsense_gyro='checkpoints/deepsense_sub_gyro/training',
                    ckpt_deepsense_gyro_imag='checkpoints/deepsense_sub_gyro_imag/training',
                    ckpt_deepsense_gps='checkpoints/deepsense_sub_gps/training'):
    initial_bias = np.log(class_weight[0] / class_weight[1])

    model = deepsense_model(input_shape, initial_bias, stacking=stacking, freeze=freeze,
                            ckpt_deepsense_acc=ckpt_deepsense_acc,
                            ckpt_deepsense_acc_imag=ckpt_deepsense_acc_imag, ckpt_deepsense_gyro=ckpt_deepsense_gyro,
                            ckpt_deepsense_gyro_imag=ckpt_deepsense_gyro_imag, ckpt_deepsense_gps=ckpt_deepsense_gps)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = ['accuracy', tf.keras.metrics.TrueNegatives(name='tn'),
               tf.keras.metrics.FalsePositives(name='fp'),
               tf.keras.metrics.FalseNegatives(name='fn'), tf.keras.metrics.TruePositives(name='tp'),
               tf.keras.metrics.AUC(curve='roc', from_logits=False, name='aucroc'),
               tf.keras.metrics.AUC(curve='PR', from_logits=False, name='aucpr'),
               TSS(), tf.keras.metrics.SensitivityAtSpecificity(0.96, name='sas')
               ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    try:
        model.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_deepsense)))
    except:
        print('There is no existing checkpoint')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_deepsense,
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


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    dir = 'Ride_Data'
    ckpt_deepsense = 'checkpoints/deepsense/training'
    ckpt_deepsense_acc = 'checkpoints/deepsense_sub_acc/training'
    ckpt_deepsense_acc_imag = 'checkpoints/deepsense_sub_acc_imag/training'
    ckpt_deepsense_gyro = 'checkpoints/deepsense_sub_gyro/training'
    ckpt_deepsense_gyro_imag = 'checkpoints/deepsense_sub_gyro_imag/training'
    ckpt_deepsense_gps = 'checkpoints/deepsense_sub_gps/training'
    region = 'Berlin'
    bucket_size = 100
    batch_size = 128
    in_memory_flag = True
    num_epochs = 100
    patience = 10
    window_size = 5
    slices = 20
    class_counts_file = os.path.join(dir, 'class_counts.csv')
    transpose_flag = False
    stacking = True
    freeze = True
    cache_dir = None
    input_shape = (None, window_size, slices, 14)

    train_ds, val_ds, test_ds, class_weight = load_data(dir, region, input_shape=input_shape, batch_size=batch_size,
                                                        in_memory_flag=in_memory_flag, transpose_flag=transpose_flag,
                                                        class_counts_file=class_counts_file, cache_dir=cache_dir)

    train_submodels(train_ds, val_ds, test_ds, class_weight, num_epochs=num_epochs, patience=patience,
                    input_shape=input_shape,
                    ckpt_deepsense_acc=ckpt_deepsense_acc, ckpt_deepsense_acc_imag=ckpt_deepsense_acc_imag,
                    ckpt_deepsense_gyro=ckpt_deepsense_gyro, ckpt_deepsense_gyro_imag=ckpt_deepsense_gyro_imag,
                    ckpt_deepsense_gps=ckpt_deepsense_gps)

    train_deepsense(train_ds, val_ds, test_ds, class_weight, num_epochs=num_epochs, patience=patience,
                    input_shape=input_shape, stacking=stacking, freeze=freeze,
                    ckpt_deepsense=ckpt_deepsense, ckpt_deepsense_acc=ckpt_deepsense_acc,
                    ckpt_deepsense_acc_imag=ckpt_deepsense_acc_imag, ckpt_deepsense_gyro=ckpt_deepsense_gyro,
                    ckpt_deepsense_gyro_imag=ckpt_deepsense_gyro_imag, ckpt_deepsense_gps=ckpt_deepsense_gps
                    )
