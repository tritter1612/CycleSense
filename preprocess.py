import os
import logging
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import datetime as dt
import multiprocessing as mp
from tqdm.auto import tqdm
from sklearn.preprocessing import MaxAbsScaler
from functools import partial
import joblib

from data_loader import create_ds, set_input_shape_global
from gan import init_gan, train_gan


def sort_timestamps_inner(file):
    df = pd.read_csv(file)
    df.sort_values(['timeStamp'], axis=0, ascending=True, inplace=True, kind='merge')
    df.to_csv(os.path.join(file), ',', index=False)


def sort_timestamps(dir, region='Berlin', pbar=None):
    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(sort_timestamps_inner, file_list)

    pbar.update(1) if pbar is not None else print()


def remove_invalid_rides_inner(file):
    df = pd.read_csv(file)

    df_cp = df.copy(deep=True)
    df_cp['timeStamp'] = df_cp['timeStamp'].diff()

    breakpoints = np.where((df_cp['timeStamp'] > 6000).to_numpy())

    df_cp.dropna(inplace=True, axis=0)

    if len(df_cp) == 0 or len(breakpoints[0]) > 0:
        # remove rides where one col is completely empty or timestamp interval is too long
        os.remove(file)


def remove_invalid_rides(dir, region='Berlin', pbar=None):
    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))

        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(remove_invalid_rides_inner, file_list)

    pbar.update(1) if pbar is not None else print()


def remove_sensor_values_from_gps_timestamps_inner(lin_acc_flag, file):
    # for android data remove accelerometer and gyroscope sensor data from gps measurements as timestamps is rounded to seconds and order is not restorable

    if os.path.splitext(file)[0][-1] == 'a':
        df = pd.read_csv(file)
        df_cp = df.copy(deep=True)
        df_cp = df_cp[['lat', 'lon', 'acc']].dropna()
        df_cp = df.iloc[df_cp.index.values].copy(True)
        df_cp[['X', 'Y', 'Z', 'a', 'b', 'c']] = ''
        if lin_acc_flag:
            df_cp[['XL', 'YL', 'ZL']] = ''
        df.iloc[df_cp.index] = df_cp
        df.to_csv(file, ',', index=False)


def remove_sensor_values_from_gps_timestamps(dir, region='Berlin', lin_acc_flag=False, pbar=None):
    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))

        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(partial(remove_sensor_values_from_gps_timestamps_inner, lin_acc_flag), file_list)

    pbar.update(1) if pbar is not None else print()


def remove_acc_outliers_inner(lower, upper, file):
    df = pd.read_csv(file)
    arr = df[['acc']].to_numpy()

    outliers_lower = arr < lower
    outliers_upper = arr > upper

    outliers = np.logical_or(outliers_lower, outliers_upper)
    outliers_bool = np.any(outliers, axis=1)
    outlier_rows = np.where(outliers_bool)[0]
    if len(outlier_rows) > 0:
        # for accuracy outliers, set lat, lon and acc to ''
        df.loc[outlier_rows, 'lat'] = ''
        df.loc[outlier_rows, 'lon'] = ''

    df.drop(columns=['acc'], inplace=True)
    df.to_csv(file, ',', index=False)


def remove_acc_outliers(dir, region='Berlin', pbar=None):
    l = []
    split = 'train'

    for file in glob.glob(os.path.join(dir, split, region, 'VM2_*.csv')):
        df = pd.read_csv(file)

        df = df[['acc']].dropna()

        if df.shape[0] == 0:
            os.remove(file)

        else:
            l.append(df[['acc']].to_numpy())

    arr = np.concatenate(l, axis=0)

    arr = arr[:, 0]
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)

    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower = q25 - cut_off
    upper = q75 + cut_off

    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))

        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(partial(remove_acc_outliers_inner, lower, upper), file_list)

    pbar.update(1) if pbar is not None else print()


def remove_acc_column_inner(file):
    df = pd.read_csv(file)
    df.drop(columns=['acc'], inplace=True)
    df.to_csv(file, ',', index=False)


def remove_acc_column(dir, region='Berlin', pbar=None):
    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(remove_acc_column_inner, file_list)

    pbar.update(1) if pbar is not None else print()


def calc_vel_delta_inner(file):
    df = pd.read_csv(file)

    df_cp = df.copy(deep=True)
    df_cp[['lat', 'lon', 'timeStamp']] = df_cp[['lat', 'lon', 'timeStamp']].dropna().diff()

    # compute lat & lon change per second
    df_cp['lat'] = df_cp['lat'].dropna() * 1000 / df_cp['timeStamp'].dropna()
    df_cp['lon'] = df_cp['lon'].dropna() * 1000 / df_cp['timeStamp'].dropna()

    df[['lat', 'lon']] = df_cp[['lat', 'lon']]

    df.to_csv(file, ',', index=False)


def calc_vel_delta(dir, region='Berlin', pbar=None):
    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(calc_vel_delta_inner, file_list)

    pbar.update(1) if pbar is not None else print()


def linear_interpolate(lin_acc_flag, file):
    df = pd.read_csv(file)

    # convert timestamp to datetime format
    df['timeStamp'] = df['timeStamp'].apply(
        lambda x: dt.datetime.utcfromtimestamp(x / 1000))

    # set timeStamp col as pandas datetime index
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='ms')

    df = df.set_index(pd.DatetimeIndex(df['timeStamp']))

    # drop all duplicate occurrences of the labels and keep the first occurrence
    df = df[~df.index.duplicated(keep='first')]

    # interpolation of X, Y, Z, a, b, c via linear interpolation based on timestamp
    df['X'].interpolate(method='time', inplace=True)
    df['Y'].interpolate(method='time', inplace=True)
    df['Z'].interpolate(method='time', inplace=True)
    df['a'].interpolate(method='time', inplace=True)
    df['b'].interpolate(method='time', inplace=True)
    df['c'].interpolate(method='time', inplace=True)

    if os.path.splitext(file)[0][-1] == 'a' and lin_acc_flag:
        df['XL'].interpolate(method='time', inplace=True)
        df['YL'].interpolate(method='time', inplace=True)
        df['ZL'].interpolate(method='time', inplace=True)

    # interpolation of missing values via padding on the reversed df
    df.sort_index(axis=0, ascending=False, inplace=True)
    df['lat'].interpolate(method='pad', inplace=True)
    df['lon'].interpolate(method='pad', inplace=True)

    df.sort_index(axis=0, ascending=True, inplace=True)

    # convert timestamp back to unix timestamp format in milliseconds
    df['timeStamp'] = df.index.view(np.int64) // 10 ** 6

    df.to_csv(file, ',', index=False)


def equidistant_interpolate(time_interval, lin_acc_flag, file):
    df = pd.read_csv(file)

    # floor start_time so that full seconds are included in the new timestamp series (time_interval may be 50, 100, 125 or 200ms)
    # this ensures that less original data are thrown away after resampling, as GPS measurements are often at full seconds
    start_time = (df['timeStamp'].iloc[0] // time_interval) * time_interval
    end_time = df['timeStamp'].iloc[-1]

    timestamps_original = df['timeStamp'].values
    # new timestamps for equidistant resampling after linear interpolation
    timestamps_new = np.arange(start_time, end_time, time_interval)
    # throw away new timestamps that are already in the original rows
    timestamps_net_new = list(set(timestamps_new) - set(timestamps_original))

    # store which original rows to remove later, as they have no equidistant timestamp
    removables = list(set(timestamps_original) - set(timestamps_new))
    removables = [dt.datetime.utcfromtimestamp(x / 1000) for x in removables]

    df_net_new = pd.concat(
        [pd.DataFrame([timestamp_net_new], columns=['timeStamp']) for timestamp_net_new in
         timestamps_net_new], ignore_index=True)
    df = pd.concat([df, df_net_new])

    # convert timestamp to datetime format
    df['timeStamp'] = df['timeStamp'].apply(
        lambda x: dt.datetime.utcfromtimestamp(x / 1000))

    # set timeStamp col as pandas datetime index
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='ms')

    df = df.set_index(pd.DatetimeIndex(df['timeStamp']))

    # drop all duplicate occurrences of the labels and keep the first occurrence,
    # as there might be some rides with original rows with duplicate timestamps
    # note that the net new timestamp rows are after the original rows
    df = df[~df.index.duplicated(keep='first')]

    # interpolation of X, Y, Z, a, b, c via linear interpolation based on timestamp
    df['X'].interpolate(method='time', inplace=True)
    df['Y'].interpolate(method='time', inplace=True)
    df['Z'].interpolate(method='time', inplace=True)
    df['a'].interpolate(method='time', inplace=True)
    df['b'].interpolate(method='time', inplace=True)
    df['c'].interpolate(method='time', inplace=True)

    if os.path.splitext(file)[0][-1] == 'a' and lin_acc_flag:
        df['XL'].interpolate(method='time', inplace=True)
        df['YL'].interpolate(method='time', inplace=True)
        df['ZL'].interpolate(method='time', inplace=True)

    # interpolation of missing lat & lon velocity values via padding on the reversed df
    df.sort_index(axis=0, ascending=False, inplace=True)
    df['lat'].interpolate(method='pad', inplace=True)
    df['lon'].interpolate(method='pad', inplace=True)

    df.sort_index(axis=0, ascending=True, inplace=True)

    incident_list = df.loc[df['incident'] > 0]

    # Assign an incident to the nearest timestamp after equidistant interpolation
    for i in range(incident_list.shape[0]):

        found = False

        while found != True:

            idx = df.index[df.index.get_indexer([incident_list.iloc[i]['timeStamp']], method='nearest')[0]]

            if idx not in removables:
                # binary time series classification
                df.at[idx, 'incident'] = 1.0
                found = True
            else:
                df = df.drop(idx)
                removables.remove(idx)

    # remove original rows which have no equidistant timestamp
    df = df.drop(removables)

    # convert timestamp back to unix timestamp format in milliseconds
    df['timeStamp'] = df.index.view(np.int64) // 10 ** 6

    df['incident'].fillna(0, inplace=True)

    df.to_csv(file, ',', index=False)


def interpolate(dir, region='Berlin', time_interval=100, interpolation_type='equidistant', lin_acc_flag=False, pbar=None):
    for split in ['train', 'test', 'val']:

        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))

        with mp.Pool(mp.cpu_count()) as pool:

            if interpolation_type == 'linear':
                pool.map(partial(linear_interpolate, lin_acc_flag), file_list)

            elif interpolation_type == 'equidistant':
                pool.map(partial(equidistant_interpolate, time_interval, lin_acc_flag), file_list)

            else:
                print('interpolation_type is incorrect')
                return

    pbar.update(1) if pbar is not None else print()


def remove_vel_outliers_inner(lower, upper, file):
    df = pd.read_csv(file)

    arr = df[['lat', 'lon']].to_numpy()

    outliers_lower = arr < lower
    outliers_upper = arr > upper

    outliers = np.logical_or(outliers_lower, outliers_upper)
    outliers_bool = np.any(outliers, axis=1)
    outlier_rows = np.where(outliers_bool)[0]

    if len(outlier_rows) > 0:
        df = df.drop(outlier_rows)
        df.to_csv(file, ',', index=False)


def remove_vel_outliers(dir, region='Berlin', pbar=None):
    l = []

    for file in glob.glob(os.path.join(dir, 'train', region, 'VM2_*.csv')):

        df = pd.read_csv(file)

        df = df.dropna()

        if df.shape[0] == 0:
            os.remove(file)

        else:
            l.append(df[['lat', 'lon']].to_numpy())

    arr = np.concatenate(l, axis=0)

    # print('lat lon data max: {}'.format(np.max(arr, axis=0)))
    # print('lat lon data min: {}'.format(np.min(arr, axis=0)))

    # arr = arr[:, :]
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)

    iqr = q75 - q25
    cut_off = iqr * 3
    lower = q25 - cut_off
    upper = q75 + cut_off

    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(partial(remove_vel_outliers_inner, lower, upper), file_list)

    pbar.update(1) if pbar is not None else print()


def remove_empty_rows_inner(file):
    df = pd.read_csv(file)
    df.dropna(inplace=True, axis=0)

    if len(df) != 0:
        df.to_csv(file, ',', index=False)
    else:
        os.remove(file)


def remove_empty_rows(dir, region='Berlin', pbar=None):
    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(remove_empty_rows_inner, file_list)

    pbar.update(1) if pbar is not None else print()


def scale_inner(scaler_maxabs, lin_acc_flag, file):
    df = pd.read_csv(file)
    if lin_acc_flag:
        df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c','XL','YL','ZL']] = scaler_maxabs.transform(
            df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c','XL','YL','ZL']])
        # change order of features and remove timestamp column
        df[['X', 'Y', 'Z', 'a', 'b', 'c', 'XL', 'YL', 'ZL', 'lat', 'lon', 'incident']].to_csv(file, ',', index=False)
    else:
        df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c']] = scaler_maxabs.transform(
            df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c']])
        # change order of features and remove timestamp column
        df[['X', 'Y', 'Z', 'a', 'b', 'c', 'lat', 'lon', 'incident']].to_csv(file, ',', index=False)


def scale(dir, region='Berlin', lin_acc_flag=False, pbar=None):
    scaler_maxabs = MaxAbsScaler()

    split = 'train'

    scaler_file = os.path.join(dir, 'scaler.save')

    if os.path.isfile(scaler_file):
        scaler_maxabs = joblib.load(scaler_file)
    else:

        for file in glob.glob(os.path.join(dir, split, region, 'VM2_*.csv')):
            df = pd.read_csv(file)

            df.fillna(0, inplace=True)

            if lin_acc_flag:
                scaler_maxabs.partial_fit(df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c', 'XL', 'YL', 'ZL']])
            else:
                scaler_maxabs.partial_fit(df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c']])
        print(scaler_maxabs.max_abs_)
        joblib.dump(scaler_maxabs, os.path.join(dir, 'scaler.save'))

    for split in ['train', 'test', 'val']:
        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))

        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(partial(scale_inner, scaler_maxabs, lin_acc_flag), file_list)

    pbar.update(1) if pbar is not None else print()


def create_buckets(dir, region='Berlin', in_memory_flag=True, window_size=5, slices=20, class_counts_file='class_counts.csv', pbar=None):
    class_counts_df = pd.DataFrame()

    for split in ['train', 'test', 'val']:

        file_list = glob.glob(os.path.join(dir, split, region, 'VM2_*.csv'))

        buckets_dict, buckets_list = {}, []

        pos_counter, neg_counter = 0, 0

        for file in file_list:

            arr = np.genfromtxt(file, delimiter=',', skip_header=True)

            # remove first and last 60 measurements of a ride
            try:
                arr = arr[60:-60, :]
            except:
                print('not enough data points to remove')

            try:

                range = (arr.shape[0] // (window_size * slices)) * window_size * slices

                arr = np.reshape(arr[:range, :], (int(range / (window_size * slices)), slices, window_size, arr.shape[1]))
                arr = np.transpose(arr, axes=(0, 2, 1, 3))

                labels = np.reshape(arr[:, :, :, -1], (arr.shape[0], slices * window_size))
                labels = np.any(labels, axis=1)

                pos_counter += np.sum(labels)
                neg_counter += len(labels) - np.sum(labels)

                if in_memory_flag:
                    for i, bucket in enumerate(arr):
                        bucket[:, :, -1] = labels[i]
                        buckets_list.append(bucket)

                else:
                    for i, bucket in enumerate(arr):
                        bucket[:, :, -1] = labels[i]
                        dict_name = os.path.basename(file).replace('.csv', '') + \
                                    '_no' + str(i).zfill(5) + '_bucket_incident'
                        buckets_dict.update({dict_name: bucket})

            except:
                print(file)

            os.remove(file)

        class_counts_df['_'.join([split, region])] = [pos_counter, neg_counter]

        class_counts_df.to_csv(os.path.join(dir, class_counts_file), ',', index=False)

        os.rmdir(os.path.join(dir, split, region))

        if in_memory_flag:
            np.savez(os.path.join(dir, split, region + '.npz'), buckets_list)
        else:
            np.savez(os.path.join(dir, split, region + '.npz'), **buckets_dict)

    pbar.update(1) if pbar is not None else print()


def rotate_bucket(ride_image, axis):
    if axis == 0:
        # 180 degree rotation matrix around X axis
        R = [[1, 0, 0],
             [0, -1, 0],
             [0, 0, -1]]

    elif axis == 1:
        # 180 degree rotation matrix around Y axis
        R = [[-1, 0, 0],
             [0, 1, 0],
             [0, 0, -1]]

    elif axis == 2:
        # 180 degree rotation matrix around Z axis
        R = [[-1, 0, 0],
             [0, -1, 0],
             [0, 0, 1]]

    else:
        return None

    ride_image_acc = ride_image[:, :, :3]
    ride_image_gyro = ride_image[:, :, 3:6]

    ride_image_acc_rotated = np.matmul(ride_image_acc, R)
    ride_image_gyro_rotated = np.matmul(ride_image_gyro, R)

    ride_image_rotated = np.concatenate((ride_image_acc_rotated, ride_image_gyro_rotated, ride_image[:, :, 6:]), axis=2)

    return ride_image_rotated


def augment_data_inner(dir, region, rotation_flag, files):

    ride_data_dict = {}
    data_loaded = np.load(os.path.join(dir, 'train', region + '.npz'))
    pos_counter = 0

    for file in files:

        ride_image = data_loaded[file]

        ride_data_dict.update({file: ride_image})

        if rotation_flag:

            if np.any(ride_image[:, :, -1]) > 0:
                dict_name_rotated_X = file.replace('_bucket_incident', '') + '_rotated_X_bucket_incident'
                dict_name_rotated_Y = file.replace('_bucket_incident', '') + '_rotated_Y_bucket_incident'
                dict_name_rotated_Z = file.replace('_bucket_incident', '') + '_rotated_Z_bucket_incident'

                ride_image_rotated_X = rotate_bucket(ride_image, axis=0)
                ride_image_rotated_Y = rotate_bucket(ride_image, axis=1)
                ride_image_rotated_Z = rotate_bucket(ride_image, axis=2)

                ride_data_dict.update({dict_name_rotated_X: ride_image_rotated_X})
                ride_data_dict.update({dict_name_rotated_Y: ride_image_rotated_Y})
                ride_data_dict.update({dict_name_rotated_Z: ride_image_rotated_Z})
                pos_counter += 3

    return ride_data_dict, pos_counter


def augment_data(dir, region='Berlin', in_memory_flag=True, rotation_flag=False, gan_flag=False, num_epochs=1000,
                 batch_size=128, latent_dim=100, input_shape=(None, 5, 20, 8), class_counts_file='class_counts.csv', gan_checkpoint_dir='gan_checkpoints', pbar=None):

    if rotation_flag or gan_flag:

        class_counts_df = pd.read_csv(os.path.join(dir, class_counts_file))

        pos_counter, neg_counter = class_counts_df['_'.join(['train', region])]

        if in_memory_flag:
            data_loaded = np.load(os.path.join(dir, 'train', region + '.npz'))
            data = data_loaded['arr_0']

            if rotation_flag:

                ride_images_list = [ride_image for ride_image in data]

                for ride_image in data:

                    if np.any(ride_image[:, :, -1]) > 0:
                        ride_image_rotated_X = rotate_bucket(ride_image, axis=0)
                        ride_image_rotated_Y = rotate_bucket(ride_image, axis=1)
                        ride_image_rotated_Z = rotate_bucket(ride_image, axis=2)

                        ride_images_list.append(ride_image_rotated_X)
                        ride_images_list.append(ride_image_rotated_Y)
                        ride_images_list.append(ride_image_rotated_Z)
                        pos_counter += 3

            if gan_flag:

                generator, discriminator = init_gan(gan_checkpoint_dir, batch_size, latent_dim)

                set_input_shape_global(input_shape)

                ds, pos_counter, neg_counter = create_ds(dir, region, 'train', batch_size=batch_size, in_memory_flag=in_memory_flag, count=True,
                                                         class_counts_file=class_counts_file, filter_fn=lambda x, y: y[0] == 1)

                try:
                    generator.load_weights(os.path.join(gan_checkpoint_dir, 'generator'))
                    print('weights have been loaded from {}'.format(gan_checkpoint_dir))
                except:
                    print('train new gan model')
                    generator, discriminator = train_gan(ds, num_epochs)

                factor = 0.1
                num_examples_to_generate = int((neg_counter - pos_counter) * factor)
                generated_buckets = generator(tf.random.normal([num_examples_to_generate, latent_dim]), training=False)
                generated_buckets = tf.concat([generated_buckets, tf.ones_like(generated_buckets)[:, :, :, :1]], axis=3)

                data = tf.concat([tf.cast(data, tf.float32), generated_buckets], axis=0)
                data = tf.random.shuffle(data)
                pos_counter += num_examples_to_generate

            np.savez(os.path.join(dir, 'train', region + '.npz'), data)

        else:

            data_loaded = np.load(os.path.join(dir, 'train', region + '.npz'))

            file_list_splits = np.array_split(data_loaded.files, mp.cpu_count())

            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(partial(augment_data_inner, dir, region, rotation_flag), file_list_splits)

            ride_data_dict = {}
            for ride_data_dict_local, pos_counter_local in results:
                ride_data_dict.update(ride_data_dict_local)
                pos_counter += pos_counter_local

            if gan_flag:

                generator, discriminator = init_gan(gan_checkpoint_dir, batch_size, latent_dim)

                set_input_shape_global(input_shape)

                ds, pos_counter, neg_counter = create_ds(dir, region, 'train', batch_size=batch_size, in_memory_flag=in_memory_flag, count=True,
                                                         class_counts_file=class_counts_file, filter_fn=lambda x, y: y == 1)

                try:
                    generator.load_weights(os.path.join(gan_checkpoint_dir, 'generator'))
                    print('weights have been loaded from {}'.format(gan_checkpoint_dir))
                except:
                    print('train new gan model')
                    generator, discriminator = train_gan(ds, num_epochs)

                factor = 0.1
                num_examples_to_generate = int((neg_counter - pos_counter) * factor)
                generated_buckets = generator(tf.random.normal([num_examples_to_generate, latent_dim]),
                                              training=False)

                generated_buckets = tf.concat([generated_buckets, tf.ones_like(generated_buckets)[:, :, :, :1]], axis=3)

                generated_dict = {'generated_bucket' + '_no' + str(i).zfill(15) + '_bucket_incident.csv':
                                      generated_bucket for i, generated_bucket in enumerate(generated_buckets)}

                pos_counter += num_examples_to_generate

                ride_data_dict.update(generated_dict)

            np.savez(os.path.join(dir, 'train', region + '.npz'), **ride_data_dict)

        class_counts_df['_'.join(['train', region])] = [pos_counter, neg_counter]
        class_counts_df.to_csv(os.path.join(dir, class_counts_file), ',', index=False)

    pbar.update(1) if pbar is not None else print()


def fourier_transform_off_memory(dir, split, region, window_size, slices, file_list):
    ride_data_dict = {}

    data_loaded = np.load(os.path.join(dir, split, region + '.npz'))

    for file in file_list:
        ride_data = data_loaded[file]
        label = ride_data[:, :, -1:]

        gps = ride_data[:, :, -3:-1]
        ride_data_transformed = np.fft.fft(ride_data[:, :, :-3], axis=0)

        data_transformed_real = np.real(ride_data_transformed)
        data_transformed_imag = np.imag(ride_data_transformed)

        ride_data_transformed = np.concatenate(
            (data_transformed_real, data_transformed_imag, gps, label), axis=2)

        ride_data_dict.update({file: ride_data_transformed})

    return ride_data_dict


def fourier_transform(dir, region='Berlin', in_memory_flag=True, fourier_transform_flag=False, window_size=5, slices=20, pbar=None):

    if fourier_transform_flag:

        for split in ['train', 'test', 'val']:

            ride_data_dict = {}

            if in_memory_flag:
                data_loaded = np.load(os.path.join(dir, split, region + '.npz'))
                data = data_loaded['arr_0']

                label = data[:, :, :, -1:]

                gps = data[:, :, :,-3:-1]
                data_transformed = np.fft.fft(data[:, :, :, :-3], axis=1)
                data_transformed_real = np.real(data_transformed)
                data_transformed_imag = np.imag(data_transformed)

                data_transformed = np.concatenate((data_transformed_real, data_transformed_imag, gps, label), axis=3)


            else:
                data_loaded = np.load(os.path.join(dir, split, region + '.npz'))
                file_list_splits = np.array_split(data_loaded.files, mp.cpu_count())

                with mp.Pool(mp.cpu_count()) as pool:
                    results = pool.map(
                        partial(fourier_transform_off_memory, dir, split, region, window_size, slices),
                        file_list_splits)
                    ride_data_dict = {}
                    for result in results:
                        ride_data_dict.update(result)

            if in_memory_flag:
                np.savez(os.path.join(dir, split, region + '.npz'), data_transformed)
            else:
                np.savez(os.path.join(dir, split, region + '.npz'), **ride_data_dict)

    pbar.update(1) if pbar is not None else print()


def preprocess(dir, region='Berlin', time_interval=100, interpolation_type='equidistant', lin_acc_flag=False,
               in_memory_flag=True, window_size=5, slices=20, fourier_transform_flag=False, rotation_flag=False,
               gan_flag=False, num_epochs=1000, batch_size=128, latent_dim=100, input_shape=(None, 5, 20, 8),
               class_counts_file='class_counts.csv', gan_checkpoint_dir='./gan_checkpoints'):

    with tqdm(total=12, desc='preprocess') as pbar:
        sort_timestamps(dir=dir, region=region, pbar=pbar)
        remove_invalid_rides(dir=dir, region=region, pbar=pbar)
        remove_sensor_values_from_gps_timestamps(dir=dir, region=region, lin_acc_flag=lin_acc_flag, pbar=pbar)
        remove_acc_outliers(dir=dir, region=region, pbar=pbar)
        calc_vel_delta(dir=dir, region=region, pbar=pbar)
        interpolate(dir=dir, region=region, time_interval=time_interval, interpolation_type=interpolation_type,
                    lin_acc_flag=lin_acc_flag, pbar=pbar)
        remove_vel_outliers(dir=dir, region=region, pbar=pbar)
        remove_empty_rows(dir=dir, region=region, pbar=pbar)
        scale(dir=dir, region=region, lin_acc_flag=lin_acc_flag, pbar=pbar)
        create_buckets(dir=dir, region=region, in_memory_flag=in_memory_flag, window_size=window_size,
                       slices=slices, class_counts_file=class_counts_file, pbar=pbar)

        augment_data(dir=dir, region=region, in_memory_flag=in_memory_flag, rotation_flag=rotation_flag, gan_flag=gan_flag, num_epochs=num_epochs,
                     batch_size=batch_size, latent_dim=latent_dim, input_shape=input_shape, class_counts_file=class_counts_file,
                     gan_checkpoint_dir=gan_checkpoint_dir, pbar=pbar)

        fourier_transform(dir=dir, region=region, in_memory_flag=in_memory_flag, fourier_transform_flag=fourier_transform_flag,
                          window_size=window_size, slices=slices, pbar=pbar)


if __name__ == '__main__':
    dir = '../Ride_Data'
    region = 'Berlin'
    time_interval = 100
    interpolation_type = 'equidistant'
    window_size = 5
    slices = 20
    lin_acc_flag = False
    in_memory_flag = True
    fourier_transform_flag = True
    rotation_flag = False
    gan_flag = True
    num_epochs = 1000
    batch_size = 128
    latent_dim = 100
    input_shape = (None, window_size, slices, 8 + 3 * lin_acc_flag)
    class_counts_file = 'class_counts.csv'
    gan_checkpoint_dir = 'gan_checkpoints'
    preprocess(dir=dir, region=region, time_interval=time_interval, interpolation_type=interpolation_type, lin_acc_flag=lin_acc_flag,
               in_memory_flag=in_memory_flag, window_size=window_size, slices=slices, fourier_transform_flag=fourier_transform_flag,
               rotation_flag=rotation_flag, gan_flag=gan_flag, num_epochs=num_epochs, batch_size=batch_size, latent_dim=latent_dim,
               input_shape=input_shape, class_counts_file=class_counts_file, gan_checkpoint_dir=gan_checkpoint_dir)
