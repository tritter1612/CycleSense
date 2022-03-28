import os
import sys
import argparse as arg
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, RNN, GRUCell, StackedRNNCells, Reshape, BatchNormalization, \
    ReLU, Dropout, add, Input
from sklearn.metrics import confusion_matrix

from data_loader import load_data
from metrics import TSS

tf.get_logger().setLevel(logging.ERROR)


def submodel_acc(input_shape=(None, 5, 20, 14), output_bias=None):
    '''
    Definition of the cyclesense accelerometer submodel.
    @param input_shape: bucket shape
    @param output_bias: output bias initialization
    @return: cyclesense accelerometer submodel
    '''
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


def submodel_acc_imag(input_shape=(None, 5, 20, 14), output_bias=None, lin_acc_flag=False):
    '''
    Definition of the cyclesense accelerometer imaginary submodel.
    @param input_shape: bucket shape
    @param output_bias: output bias initialization
    @param lin_acc_flag: whether the linear accelerometer data was exported, too
    @return: cyclesense acclerometer imaginary submodel
    '''
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    acc = x[:, :, :, 6:9] if not lin_acc_flag else x[:, :, :, 9:12]

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
    '''
    Definition of the cyclesense gyroscope submodel.
    @param input_shape: bucket shape
    @param output_bias: output bias initialization
    @return: cyclesense gyro submodel
    '''
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


def submodel_gyro_imag(input_shape=(None, 5, 20, 14), output_bias=None, lin_acc_flag=False):
    '''
    Definition of the cyclesense gyroscope imaginary submodel.
    @param input_shape: bucket shape
    @param output_bias: output bias initialization
    @param lin_acc_flag: whether the linear accelerometer data was exported, too
    @return: cyclesense gyroscope imaginary submodel
    '''
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    gyro = x[:, :, :, 9:12] if not lin_acc_flag else x[:, :, :, 12:15]

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


def submodel_gps(input_shape=(None, 5, 20, 14), output_bias=None, fourier_transform_flag=True, lin_acc_flag=False):
    '''
    Definition of the cyclesense gps submodel.
    @param input_shape: bucket shape
    @param output_bias: output bias initialization
    @param fourier_transform_flag: whether fourier transform was applied on the data
    @param lin_acc_flag: whether the linear accelerometer data was exported, too
    @return: cyclesense gps submodel
    '''
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    if fourier_transform_flag:
        if lin_acc_flag:
            gps = x[:, :, :, 18:]
        else:
            gps = x[:, :, :, 12:]
    else:
        if lin_acc_flag:
            gps = x[:, :, :, 9:]
        else:
            gps = x[:, :, :, 6:]

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


def submodel_linacc(input_shape=(None, 5, 20, 20), output_bias=None):
    '''
    Definition of the cyclesense linacc submodel.
    @param input_shape: bucket shape
    @param output_bias: output bias initialization
    @return: cyclesense linacc submodel
    '''
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    linacc = x[:, :, :, 6:9]

    linacc = linacc[:, :, :, :, tf.newaxis]

    linacc_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape)(linacc)
    linacc_conv1 = BatchNormalization()(linacc_conv1)
    linacc_conv1 = ReLU()(linacc_conv1)
    linacc_conv1 = Dropout(0.5)(linacc_conv1)

    linacc_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(linacc_conv1)
    linacc_conv2 = BatchNormalization()(linacc_conv2)
    linacc_conv2 = ReLU()(linacc_conv2)
    linacc_conv2 = Dropout(0.5)(linacc_conv2)

    linacc_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(linacc_conv2)
    linacc_conv3 = BatchNormalization()(linacc_conv3)
    linacc_conv3 = ReLU()(linacc_conv3)

    linacc_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid')(linacc)
    linacc_shortcut = BatchNormalization()(linacc_shortcut)
    linacc_shortcut = ReLU()(linacc_shortcut)

    linacc = add([linacc_conv3, linacc_shortcut])

    linacc_fc = Flatten()(linacc)
    linacc_fc = Dense(1, activation='sigmoid', bias_initializer=output_bias)(linacc_fc)

    model = Model(x, linacc_fc)

    return model


def submodel_linacc_imag(input_shape=(None, 5, 20, 20), output_bias=None):
    '''
    Definition of the cyclesense linacc imaginary submodel.
    @param input_shape: bucket shape
    @param output_bias: output bias initialization
    @return: cyclesense linacc imaginary submodel
    '''
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    linacc = x[:, :, :, 15:18]

    linacc = linacc[:, :, :, :, tf.newaxis]

    linacc_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape)(linacc)
    linacc_conv1 = BatchNormalization()(linacc_conv1)
    linacc_conv1 = ReLU()(linacc_conv1)
    linacc_conv1 = Dropout(0.5)(linacc_conv1)

    linacc_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(linacc_conv1)
    linacc_conv2 = BatchNormalization()(linacc_conv2)
    linacc_conv2 = ReLU()(linacc_conv2)
    linacc_conv2 = Dropout(0.5)(linacc_conv2)

    linacc_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same')(linacc_conv2)
    linacc_conv3 = BatchNormalization()(linacc_conv3)
    linacc_conv3 = ReLU()(linacc_conv3)

    linacc_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid')(linacc)
    linacc_shortcut = BatchNormalization()(linacc_shortcut)
    linacc_shortcut = ReLU()(linacc_shortcut)

    linacc = add([linacc_conv3, linacc_shortcut])

    linacc_fc = Flatten()(linacc)
    linacc_fc = Dense(1, activation='sigmoid', bias_initializer=output_bias)(linacc_fc)

    model = Model(x, linacc_fc)

    return model


def cyclesense_model(input_shape=(None, 5, 20, 14), output_bias=None, stacking=True, freeze=True,
                    fourier_transform_flag=True, lin_acc_flag=False,
                    ckpt_cyclesense_acc='checkpoints/cyclesense_sub_acc/training',
                    ckpt_cyclesense_acc_imag='checkpoints/cyclesense_sub_acc_imag/training',
                    ckpt_cyclesense_gyro='checkpoints/cyclesense_sub_gyro/training',
                    ckpt_cyclesense_gyro_imag='checkpoints/cyclesense_sub_gyro_imag/training',
                    ckpt_cyclesense_linacc='checkpoints/cyclesense_sub_linacc/training',
                    ckpt_cyclesense_linacc_imag='checkpoints/cyclesense_sub_linacc_imag/training',
                    ckpt_cyclesense_gps='checkpoints/cyclesense_sub_gps/training'):
    '''
    Definition of the cyclesense model.
    @param input_shape: bucket shape
    @param stacking: whether to train with stracking
    @param freeze: whether to freeze the submodel weights
    @param fourier_transform_flag: whether fourier transform was applied on the data
    @param lin_acc_flag: whether the linear accelerometer data was exported, too
    @param ckpt_cyclesense_acc: checkpoint path cyclesense accelerometer submodel
    @param ckpt_cyclesense_acc_imag: checkpoint path cyclesense accelerometer imaginary submodel
    @param ckpt_cyclesense_gyro: checkpoint path cyclesense gyroscope submodel
    @param ckpt_cyclesense_gyro_imag: checkpoint path cyclesense gyroscope imaginary submodel
    @param ckpt_cyclesense_linacc: checkpoint path cyclesense linacc submodel
    @param ckpt_cyclesense_linacc_imag: checkpoint path cyclesense linacc imaginary submodel
    @param ckpt_cyclesense_gps: checkpoint path cyclesense gps submodel
    '''


    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    x = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))

    if lin_acc_flag:
        if fourier_transform_flag:
            acc_real = x[:, :, :, 0:3]
            gyro_real = x[:, :, :, 3:6]
            linacc_real = x[:, :, :, 6:9]
            acc_imag = x[:, :, :, 9:12]
            gyro_imag = x[:, :, :, 12:15]
            linacc_imag = x[:, :, :, 15:18]
            gps = x[:, :, :, 18:]
        else:
            acc_real = x[:, :, :, 0:3]
            gyro_real = x[:, :, :, 3:6]
            linacc_real = x[:, :, :, 6:9]
            gps = x[:, :, :, 9:]

    else:
        if fourier_transform_flag:
            acc_real = x[:, :, :, 0:3]
            gyro_real = x[:, :, :, 3:6]
            acc_imag = x[:, :, :, 6:9]
            gyro_imag = x[:, :, :, 9:12]
            gps = x[:, :, :, 12:]
        else:
            acc_real = x[:, :, :, 0:3]
            gyro_real = x[:, :, :, 3:6]
            gps = x[:, :, :, 6:]

    acc_real = acc_real[:, :, :, :, tf.newaxis]
    gyro_real = gyro_real[:, :, :, :, tf.newaxis]
    if lin_acc_flag:
        linacc_real = linacc_real[:, :, :, :, tf.newaxis]

    if fourier_transform_flag:
        acc_imag = acc_imag[:, :, :, :, tf.newaxis]
        gyro_imag = gyro_imag[:, :, :, :, tf.newaxis]
        if lin_acc_flag:
            linacc_imag = linacc_imag[:, :, :, :, tf.newaxis]

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

    if lin_acc_flag:
        linacc_real_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape,
                                   trainable=not freeze)(linacc_real)
        linacc_real_conv1 = BatchNormalization()(linacc_real_conv1)
        linacc_real_conv1 = ReLU()(linacc_real_conv1)
        linacc_real_conv1 = Dropout(0.5)(linacc_real_conv1)

        linacc_real_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(linacc_real_conv1)
        linacc_real_conv2 = BatchNormalization()(linacc_real_conv2)
        linacc_real_conv2 = ReLU()(linacc_real_conv2)
        linacc_real_conv2 = Dropout(0.5)(linacc_real_conv2)

        linacc_real_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(linacc_real_conv2)
        linacc_real_conv3 = BatchNormalization()(linacc_real_conv3)
        linacc_real_conv3 = ReLU()(linacc_real_conv3)

        linacc_real_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', trainable=not freeze)(linacc_real)
        linacc_real_shortcut = BatchNormalization()(linacc_real_shortcut)
        linacc_real_shortcut = ReLU()(linacc_real_shortcut)

        linacc_real = add([linacc_real_conv3, linacc_real_shortcut])

    if fourier_transform_flag:

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

        if lin_acc_flag:
            linacc_imag_conv1 = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', input_shape=input_shape,
                                       trainable=not freeze)(
                linacc_imag)
            linacc_imag_conv1 = BatchNormalization()(linacc_imag_conv1)
            linacc_imag_conv1 = ReLU()(linacc_imag_conv1)
            linacc_imag_conv1 = Dropout(0.5)(linacc_imag_conv1)

            linacc_imag_conv2 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(linacc_imag_conv1)
            linacc_imag_conv2 = BatchNormalization()(linacc_imag_conv2)
            linacc_imag_conv2 = ReLU()(linacc_imag_conv2)
            linacc_imag_conv2 = Dropout(0.5)(linacc_imag_conv2)

            linacc_imag_conv3 = Conv3D(64, kernel_size=(3, 3, 1), padding='same', trainable=not freeze)(linacc_imag_conv2)
            linacc_imag_conv3 = BatchNormalization()(linacc_imag_conv3)
            linacc_imag_conv3 = ReLU()(linacc_imag_conv3)

            linacc_imag_shortcut = Conv3D(64, kernel_size=(3, 3, 3), padding='valid', trainable=not freeze)(linacc_imag)
            linacc_imag_shortcut = BatchNormalization()(linacc_imag_shortcut)
            linacc_imag_shortcut = ReLU()(linacc_imag_shortcut)

            linacc_imag = add([linacc_imag_conv3, linacc_imag_shortcut])

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
        model_acc.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense_acc)))

        gyro_fc = Flatten()(gyro_real)
        gyro_fc = Dense(1, activation='sigmoid')(gyro_fc)

        model_gyro = Model(x, gyro_fc)
        model_gyro.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense_gyro)))

        if lin_acc_flag:
            linacc_fc = Flatten()(linacc_real)
            linacc_fc = Dense(1, activation='sigmoid')(linacc_fc)

            model_linacc = Model(x, linacc_fc)
            model_linacc.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense_linacc)))

        if fourier_transform_flag:

            acc_imag_fc = Flatten()(acc_imag)
            acc_imag_fc = Dense(1, activation='sigmoid')(acc_imag_fc)

            model_acc_imag = Model(x, acc_imag_fc)
            model_acc_imag.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense_acc_imag)))

            gyro_imag_fc = Flatten()(gyro_imag)
            gyro_imag_fc = Dense(1, activation='sigmoid')(gyro_imag_fc)

            model_gyro_imag = Model(x, gyro_imag_fc)
            model_gyro_imag.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense_gyro_imag)))


            if lin_acc_flag:
                linacc_imag_fc = Flatten()(linacc_imag)
                linacc_imag_fc = Dense(1, activation='sigmoid')(linacc_imag_fc)

                model_linacc_imag = Model(x, linacc_imag_fc)
                model_linacc_imag.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense_linacc_imag)))

        gps_fc = Flatten()(gps)
        gps_fc = Dense(1, activation='sigmoid')(gps_fc)

        model_gps = Model(x, gps_fc)
        model_gps.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense_gps)))

    if lin_acc_flag:
        if fourier_transform_flag:
            sensor = tf.concat([acc_real, acc_imag, gyro_real, gyro_imag, linacc_real, linacc_imag, gps], 3)
        else:
            sensor = tf.concat([acc_real, gyro_real, linacc_real, gps], 3)
    else:
        if fourier_transform_flag:
            sensor = tf.concat([acc_real, acc_imag, gyro_real, gyro_imag, gps], 3)
        else:
            sensor = tf.concat([acc_real, gyro_real, gps], 3)

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

    sensor = Reshape((input_shape[2] - 2, (input_shape[1] - 2) * ((2 + lin_acc_flag) * (fourier_transform_flag + 1) + 1) * 64))(sensor)

    sensor_gru1 = GRUCell(120, dropout=0.5, activation=None)
    sensor_gru2 = GRUCell(120, dropout=0.5, activation=None)
    sensor = RNN(StackedRNNCells([sensor_gru1, sensor_gru2]), return_sequences=True)(sensor)

    sensor = tf.math.reduce_mean(sensor, axis=1, keepdims=False)

    sensor = Dense(1, activation='sigmoid', bias_initializer=output_bias)(sensor)

    model = Model(x, sensor)

    return model


def train_submodels(train_ds, val_ds, test_ds, class_weight, num_epochs=10, patience=1, input_shape=(None, 5, 20, 14),
                    fourier_transform_flag=True, lin_acc_flag=False,
                    ckpt_cyclesense_acc='checkpoints/cyclesense_sub_acc/training',
                    ckpt_cyclesense_acc_imag='checkpoints/cyclesense_sub_acc_imag/training',
                    ckpt_cyclesense_gyro='checkpoints/cyclesense_sub_gyro/training',
                    ckpt_cyclesense_gyro_imag='checkpoints/cyclesense_sub_gyro_imag/training',
                    ckpt_cyclesense_linacc='checkpoints/cyclesense_sub_linacc/training',
                    ckpt_cyclesense_linacc_imag='checkpoints/cyclesense_sub_linacc_imag/training',
                    ckpt_cyclesense_gps='checkpoints/cyclesense_sub_gps/training'
                    ):
    '''
    Training method for cyclesense submodels model.
    @param train_ds: training dataset
    @param val_ds: validation dataset
    @param test_ds: test dataset
    @param class_weight: class weight dictionary for weighted loss function
    @param num_epochs: number of training epochs
    @param patience: patience
    @param input_shape: bucket shape
    @param fourier_transform_flag: whether fourier transform was applied on the data
    @param lin_acc_flag: whether the linear accelerometer data was exported, too
    @param ckpt_cyclesense_acc: checkpoint path cyclesense accelerometer submodel
    @param ckpt_cyclesense_acc_imag: checkpoint path cyclesense accelerometer imaginary submodel
    @param ckpt_cyclesense_gyro: checkpoint path cyclesense gyroscope submodel
    @param ckpt_cyclesense_gyro_imag: checkpoint path cyclesense gyroscope imaginary submodel
    @param ckpt_cyclesense_linacc: checkpoint path cyclesense linacc submodel
    @param ckpt_cyclesense_linacc_imag: checkpoint path cyclesense linacc imaginary submodel
    @param ckpt_cyclesense_gps: checkpoint path cyclesense gps submodel
    '''
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

    if lin_acc_flag:
        submodels = [(submodel_acc(input_shape, initial_bias), ckpt_cyclesense_acc),
                     (submodel_acc_imag(input_shape, initial_bias, lin_acc_flag), ckpt_cyclesense_acc_imag),
                     (submodel_gyro(input_shape, initial_bias), ckpt_cyclesense_gyro),
                     (submodel_gyro_imag(input_shape, initial_bias, lin_acc_flag), ckpt_cyclesense_gyro_imag),
                     (submodel_linacc(input_shape, initial_bias), ckpt_cyclesense_linacc),
                     (submodel_linacc_imag(input_shape, initial_bias), ckpt_cyclesense_linacc_imag),
                     (submodel_gps(input_shape, initial_bias, fourier_transform_flag, lin_acc_flag), ckpt_cyclesense_gps)]

    else:
        submodels = [(submodel_acc(input_shape, initial_bias), ckpt_cyclesense_acc),
                     (submodel_acc_imag(input_shape, initial_bias, lin_acc_flag), ckpt_cyclesense_acc_imag),
                     (submodel_gyro(input_shape, initial_bias), ckpt_cyclesense_gyro),
                     (submodel_gyro_imag(input_shape, initial_bias, lin_acc_flag), ckpt_cyclesense_gyro_imag),
                     (submodel_gps(input_shape, initial_bias, fourier_transform_flag, lin_acc_flag), ckpt_cyclesense_gps)]


    for model, ckpt_cyclesense_sub in submodels:

        metrics = ['accuracy',
                   tf.keras.metrics.TrueNegatives(name='tn'),
                   tf.keras.metrics.FalsePositives(name='fp'),
                   tf.keras.metrics.FalseNegatives(name='fn'),
                   tf.keras.metrics.TruePositives(name='tp'),
                   tf.keras.metrics.AUC(curve='roc', from_logits=False, name='aucroc')]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        try:
            model.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense_sub)))
        except:
            print('There is no existing checkpoint')

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_cyclesense_sub,
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


def train_cyclesense(train_ds, val_ds, test_ds, class_weight, num_epochs=10, patience=1, input_shape=(None, 5, 20, 14),
                    stacking=False, freeze=False, fourier_transform_flag=True, lin_acc_flag=False,
                    ckpt_cyclesense='checkpoints/cyclesense/training',
                    ckpt_cyclesense_acc='checkpoints/cyclesense_sub_acc/training',
                    ckpt_cyclesense_acc_imag='checkpoints/cyclesense_sub_acc_imag/training',
                    ckpt_cyclesense_gyro='checkpoints/cyclesense_sub_gyro/training',
                    ckpt_cyclesense_gyro_imag='checkpoints/cyclesense_sub_gyro_imag/training',
                    ckpt_cyclesense_linacc='checkpoints/cyclesense_sub_linacc/training',
                    ckpt_cyclesense_linacc_imag='checkpoints/cyclesense_sub_linacc_imag/training',
                    ckpt_cyclesense_gps='checkpoints/cyclesense_sub_gps/training'):
    '''
    Training method for cyclesense model.
    @param train_ds: training dataset
    @param val_ds: validation dataset
    @param test_ds: test dataset
    @param class_weight: class weight dictionary for weighted loss function
    @param num_epochs: number of training epochs
    @param patience: patience
    @param input_shape: bucket shape
    @param stacking: whether to train with stracking
    @param freeze: whether to freeze the submodel weights
    @param fourier_transform_flag: whether fourier transform was applied on the data
    @param lin_acc_flag: whether the linear accelerometer data was exported, too
    @param ckpt_cyclesense: checkpoint path cyclesense model
    @param ckpt_cyclesense_acc: checkpoint path cyclesense accelerometer submodel
    @param ckpt_cyclesense_acc_imag: checkpoint path cyclesense accelerometer imaginary submodel
    @param ckpt_cyclesense_gyro: checkpoint path cyclesense gyroscope submodel
    @param ckpt_cyclesense_gyro_imag: checkpoint path cyclesense gyroscope imaginary submodel
    @param ckpt_cyclesense_linacc: checkpoint path cyclesense linacc submodel
    @param ckpt_cyclesense_linacc_imag: checkpoint path cyclesense linacc imaginary submodel
    @param ckpt_cyclesense_gps: checkpoint path cyclesense gps submodel
    '''
    initial_bias = np.log(class_weight[0] / class_weight[1])

    model = cyclesense_model(input_shape, initial_bias, stacking=stacking, freeze=freeze,
                            fourier_transform_flag=fourier_transform_flag, lin_acc_flag=lin_acc_flag,
                            ckpt_cyclesense_acc=ckpt_cyclesense_acc,
                            ckpt_cyclesense_acc_imag=ckpt_cyclesense_acc_imag, ckpt_cyclesense_gyro=ckpt_cyclesense_gyro,
                            ckpt_cyclesense_gyro_imag=ckpt_cyclesense_gyro_imag, ckpt_cyclesense_linacc=ckpt_cyclesense_linacc,
                            ckpt_cyclesense_linacc_imag=ckpt_cyclesense_linacc_imag, ckpt_cyclesense_gps=ckpt_cyclesense_gps)

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
        model.load_weights(tf.train.latest_checkpoint(os.path.dirname(ckpt_cyclesense)))
    except:
        print('There is no existing checkpoint')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_cyclesense,
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
    parser = arg.ArgumentParser(description='cyclesense')
    parser.add_argument('dir', metavar='<directory>', type=str, help='path to the data directory')
    parser.add_argument('--region', metavar='<region>', type=str, help='target region', required=False, default='Berlin')
    parser.add_argument('--ckpt_cyclesense', metavar='<directory>', type=str, help='checkpoint path cyclesense model', required=False, default='checkpoints/cyclesense/training')
    parser.add_argument('--ckpt_cyclesense_acc', metavar='directory>', type=str, help='checkpoint path cyclesense accelerometer submodel', required=False, default='checkpoints/cyclesense_sub_acc/training')
    parser.add_argument('--ckpt_cyclesense_acc_imag', metavar='<directory>', type=str, help='checkpoint path cyclesense accelerometer imaginary submodel', required=False, default='checkpoints/cyclesense_sub_acc_imag/training')
    parser.add_argument('--ckpt_cyclesense_gyro', metavar='<directory>', type=str, help='checkpoint path cyclesense gyroscope submodel', required=False, default='checkpoints/cyclesense_sub_gyro/training')
    parser.add_argument('--ckpt_cyclesense_gyro_imag', metavar='<directory>', type=str, help='checkpoint path cyclesense gyroscope imaginary submodel', required=False, default='checkpoints/cyclesense_sub_gyro_imag/training')
    parser.add_argument('--ckpt_cyclesense_linacc', metavar='<directory>', type=str, help='checkpoint path cyclesense linacc submodel', required=False, default='checkpoints/cyclesense_sub_linacc/training')
    parser.add_argument('--ckpt_cyclesense_linacc_imag', metavar='<directory>', type=str, help='checkpoint path cyclesense linacc imaginary submodel', required=False, default='checkpoints/cyclesense_sub_linacc/training')
    parser.add_argument('--ckpt_cyclesense_gps', metavar='<directory>', type=str, help='checkpoint path cyclesense gps submodel', required=False, default='checkpoints/cyclesense_sub_gps/training')
    parser.add_argument('--num_epochs', metavar='<int>', type=int, help='training epochs', required=False, default=100)
    parser.add_argument('--patience', metavar='<int>', type=int, help='patience value for early stopping', required=False, default=10)
    parser.add_argument('--batch_size', metavar='<int>', type=int, help='batch size', required=False, default=128)
    parser.add_argument('--window_size', metavar='<int>', type=int, help='bucket height', required=False, default=5)
    parser.add_argument('--slices', metavar='<int>', type=int, help='bucket width', required=False, default=20)
    parser.add_argument('--lin_acc_flag', metavar='<bool>', type=bool, help='whether the linear accelerometer data was exported, too', required=False, default=False)
    parser.add_argument('--in_memory_flag', metavar='<bool>', type=bool, help='whether the data was stored in one array or not', required=False, default=True)
    parser.add_argument('--fourier_transform_flag', metavar='<bool>', type=bool, help='whether fourier transform was applied on the data', required=False, default=True)
    parser.add_argument('--stacking', metavar='<bool>', type=bool, help='whether to train with stracking', required=False, default=True)
    parser.add_argument('--freeze', metavar='<bool>', type=bool, help='whether to freeze the submodel weights', required=False, default=True)
    parser.add_argument('--class_counts_file', metavar='<file>', type=str, help='path to class counts file', required=False, default='class_counts.csv')
    parser.add_argument('--cache_dir', metavar='<directory>', type=str, help='path to cache directory', required=False, default=None)
    args = parser.parse_args()

    input_shape = (None, args.window_size, args.slices, 2 + 6 * (1 + args.fourier_transform_flag) + 3 * (1 + args.fourier_transform_flag) * args.lin_acc_flag)

    train_ds, val_ds, test_ds, class_weight = load_data(args.dir, args.region, input_shape=input_shape, batch_size=args.batch_size,
                                                        in_memory_flag=args.in_memory_flag, transpose_flag=False,
                                                        class_counts_file=args.class_counts_file, cache_dir=args.cache_dir)

    train_submodels(train_ds, val_ds, test_ds, class_weight, num_epochs=args.num_epochs, patience=args.patience,
                    input_shape=input_shape, fourier_transform_flag=args.fourier_transform_flag, lin_acc_flag=args.lin_acc_flag,
                    ckpt_cyclesense_acc=args.ckpt_cyclesense_acc, ckpt_cyclesense_acc_imag=args.ckpt_cyclesense_acc_imag,
                    ckpt_cyclesense_gyro=args.ckpt_cyclesense_gyro, ckpt_cyclesense_gyro_imag=args.ckpt_cyclesense_gyro_imag,
                    ckpt_cyclesense_linacc=args.ckpt_cyclesense_linacc, ckpt_cyclesense_linacc_imag=args.ckpt_cyclesense_linacc_imag,
                    ckpt_cyclesense_gps=args.ckpt_cyclesense_gps)

    train_cyclesense(train_ds, val_ds, test_ds, class_weight, num_epochs=args.num_epochs, patience=args.patience,
                    input_shape=input_shape, stacking=args.stacking, freeze=args.freeze,
                    fourier_transform_flag=args.fourier_transform_flag, lin_acc_flag=args.lin_acc_flag,
                    ckpt_cyclesense=args.ckpt_cyclesense, ckpt_cyclesense_acc=args.ckpt_cyclesense_acc,
                    ckpt_cyclesense_acc_imag=args.ckpt_cyclesense_acc_imag, ckpt_cyclesense_gyro=args.ckpt_cyclesense_gyro,
                    ckpt_cyclesense_gyro_imag=args.ckpt_cyclesense_gyro_imag, ckpt_cyclesense_linacc=args.ckpt_cyclesense_linacc,
                    ckpt_cyclesense_linacc_imag=args.ckpt_cyclesense_linacc_imag, ckpt_cyclesense_gps=args.ckpt_cyclesense_gps
                    )

if __name__ == '__main__':
    main(sys.argv[1:])
