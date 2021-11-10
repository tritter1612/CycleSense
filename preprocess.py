import os
import glob
import numpy as np
import pandas as pd
import datetime as dt
import multiprocessing as mp
from tqdm.auto import tqdm
from sklearn.preprocessing import MaxAbsScaler
import math
from functools import partial
import joblib


def remove_invalid_rides(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='remove invalid rides in {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):

                df = pd.read_csv(file)

                df_cp = df.copy(deep=True)
                df_cp['timeStamp'] = df_cp['timeStamp'].diff()

                breakpoints = np.where((df_cp['timeStamp'] > 6000).to_numpy())

                df_cp.dropna(inplace=True, axis=0)

                if len(df_cp) == 0 or len(breakpoints[0]) > 0:
                    # remove rides where one col is completely empty or gps timestamp interval is too long
                    os.remove(file)


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


def remove_acc_outliers(dir, target_region=None):
    l = []
    split = 'train'

    for subdir in glob.glob(os.path.join(dir, split, '[!.]*')):

        region = os.path.basename(subdir)

        if target_region is not None and target_region != region:
            continue

        for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):
            df = pd.read_csv(file)

            df = df.dropna()

            if df.shape[0] == 0:
                os.remove(file)

            else:
                l.append(df[['acc']].to_numpy())

    arr = np.concatenate(l, axis=0)
    print('data max: {}'.format(np.max(arr, axis=0)))
    print('data min: {}'.format(np.min(arr, axis=0)))

    arr = arr[:, 0]
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)

    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower = q25 - cut_off
    upper = q75 + cut_off

    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='remove accuracy outliers from {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))

            with mp.Pool(mp.cpu_count()) as pool:
                pool.map(partial(remove_acc_outliers_inner, lower, upper), file_list)


def calc_vel_delta(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='calculate vel delta on {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):
                df = pd.read_csv(file)

                df_cp = df.copy(deep=True)

                df_cp = df_cp.dropna()
                df_cp[['lat', 'lon', 'timeStamp']] = df_cp[['lat', 'lon', 'timeStamp']].diff()
                df_cp = df_cp.dropna()

                # compute lat & lon change per second
                df_cp['lat'] = df_cp['lat'] * 1000 / df_cp['timeStamp']
                df_cp['lon'] = df_cp['lon'] * 1000 / df_cp['timeStamp']

                df[['lat', 'lon']] = df_cp[['lat', 'lon']]

                df.to_csv(file, ',', index=False)


def linear_interpolate(file):
    df = pd.read_csv(file)

    # convert timestamp to datetime format
    df['timeStamp'] = df['timeStamp'].apply(
        lambda x: dt.datetime.utcfromtimestamp(x / 1000))

    # set timeStamp col as pandas datetime index
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='ms')

    df = df.set_index(pd.DatetimeIndex(df['timeStamp']))

    # drop all duplicate occurrences of the labels and keep the first occurrence
    df = df[~df.index.duplicated(keep='first')]

    # interpolation of a, b, c via linear interpolation based on timestamp
    df['a'].interpolate(method='time', inplace=True)
    df['b'].interpolate(method='time', inplace=True)
    df['c'].interpolate(method='time', inplace=True)

    df.sort_index(axis=0, ascending=False, inplace=True)

    # interpolation of missing values via padding on the reversed df
    df['lat'].interpolate(method='pad', inplace=True)
    df['lon'].interpolate(method='pad', inplace=True)

    df.sort_index(axis=0, ascending=True, inplace=True)

    # convert timestamp back to unix timestamp format in milliseconds
    df['timeStamp'] = df.index.view(np.int64) // 10 ** 6

    df.to_csv(file, ',', index=False)


def equidistant_interpolate(time_interval, file):
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

    # interpolation of missing lat & lon velocity values via padding on the reversed df
    df.sort_index(axis=0, ascending=False, inplace=True)
    df['lat'].interpolate(method='pad', inplace=True)
    df['lon'].interpolate(method='pad', inplace=True)

    df.sort_index(axis=0, ascending=True, inplace=True)

    incident_list = df.loc[df['incident'] > 0]

    for i in range(incident_list.shape[0]):

        found = False

        while found != True:

            idx = df.index[df.index.get_loc(incident_list.iloc[i]['timeStamp'], method='nearest')]

            if idx not in removables:
                df.at[idx, 'incident'] = 1.0  # TODO: preserve incident type
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


def interpolate(dir, target_region=None, time_interval=100, interpolation_type='linear'):
    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='interpolate {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))

            with mp.Pool(mp.cpu_count()) as pool:

                if interpolation_type == 'linear':
                    pool.map(linear_interpolate, file_list)

                elif interpolation_type == 'equidistant':
                    pool.map(partial(equidistant_interpolate, time_interval), file_list)

                else:
                    print('interpolation_type is incorrect')
                    return


def remove_vel_outliers(dir, target_region=None):
    l = []
    split = 'train'

    for subdir in glob.glob(os.path.join(dir, split, '[!.]*')):
        region = os.path.basename(subdir)

        if target_region is not None and target_region != region:
            continue

        for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):

            df = pd.read_csv(file)

            df = df.dropna()

            if df.shape[0] == 0:
                os.remove(file)

            else:
                l.append(df[['lat', 'lon']].to_numpy())

    arr = np.concatenate(l, axis=0)

    print('data max: {}'.format(np.max(arr, axis=0)))
    print('data min: {}'.format(np.min(arr, axis=0)))

    # arr = arr[:, :]
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)

    iqr = q75 - q25
    cut_off = iqr * 3
    lower = q25 - cut_off
    upper = q75 + cut_off

    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='remove velocity outliers from {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):

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


def remove_empty_rows(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='remove empty rows in {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):

                df = pd.read_csv(file)
                if not gps_flag:
                    df['lat'] = 0.0
                    df['lon'] = 0.0
                df.dropna(inplace=True, axis=0)

                if len(df) != 0:
                    df.to_csv(file, ',', index=False)
                else:
                    os.remove(file)


def scale(dir, target_region=None):
    scaler_maxabs = MaxAbsScaler()

    split = 'train'

    scaler_file = os.path.join(dir, 'scaler.save')

    if os.path.isfile(scaler_file):
        scaler_maxabs = joblib.load(scaler_file)
    else:
        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')), desc='fit scaler'):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):
                df = pd.read_csv(file)

                df.fillna(0, inplace=True)

                scaler_maxabs.partial_fit(df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c']])

        joblib.dump(scaler_maxabs, os.path.join(dir, 'scaler.save'))

    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')), desc='scale {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):
                df = pd.read_csv(file)
                df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c']] = scaler_maxabs.transform(
                    df[['lat', 'lon', 'X', 'Y', 'Z', 'a', 'b', 'c']])

                df.to_csv(file, ',', index=False)


def augment_data(ride_image, axis):

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


def create_buckets_inner(bucket_size, file):
    df = pd.read_csv(file)

    length = df.shape[0]
    num_splits = math.floor(length / bucket_size)
    length_new = num_splits * bucket_size

    if num_splits >= 1:

        df_splits = np.array_split(df.iloc[:length_new, :], num_splits)

        for k, df_split in enumerate(df_splits):

            if (np.any((df_split['incident'] >= 1.0).to_numpy())):
                df_split['incident'] = 1.0
                df_split.to_csv(file.replace('.csv', '') + '_no' + str(k).zfill(5) + '_bucket_incident.csv', ',',
                                index=False)
            else:
                df_split['incident'] = 0.0
                df_split.to_csv(file.replace('.csv', '') + '_no' + str(k).zfill(5) + '_bucket.csv', ',', index=False)

    os.remove(file)


def create_buckets(dir, target_region=None, bucket_size=100, in_memory_flag=True, deepsense_flag=False, fft_window=5, slices=20,
                   data_augmentation_flag=False, gps_flag=False, class_counts_file='class_counts.csv'):
    class_counts_df = pd.DataFrame()

    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='generate buckets for {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))

            ride_images_dict, ride_images_list = {}, []

            pos_counter, neg_counter = 0, 0

            if deepsense_flag:

                for file in file_list:

                    arr = np.genfromtxt(file, delimiter=',', skip_header=True)

                    try:

                        lat = arr[:, 0]
                        lat = lat[:, np.newaxis]
                        lon = arr[:, 1]
                        lon = lon[:, np.newaxis]
                        incident = arr[:, -1]
                        incident = incident[:, np.newaxis]

                        # remove lat, lon
                        arr = arr[:, 2:]

                        # remove incident
                        arr = arr[:, :-1]

                        # remove timestamp
                        arr = np.concatenate((arr[:, :3], arr[:, 4:]), axis=1)

                        n_window_splits = arr.shape[0] // fft_window
                        window_split_range = n_window_splits * fft_window

                        if n_window_splits > 0:

                            ride_images = np.stack(np.vsplit(arr[:window_split_range], n_window_splits), axis=1)
                            lat = np.stack(np.vsplit(lat[:window_split_range], n_window_splits), axis=1)
                            lon = np.stack(np.vsplit(lon[:window_split_range], n_window_splits), axis=1)
                            incident = np.stack(np.vsplit(incident[:window_split_range], n_window_splits), axis=1)

                            n_image_splits = n_window_splits // slices
                            image_split_range = n_image_splits * slices

                            if n_image_splits > 0:
                                ride_image_list = np.array_split(ride_images[:, :image_split_range, :], n_image_splits, axis=1)
                                lat_list = np.array_split(lat[:, :image_split_range, :], n_image_splits, axis=1)
                                lon_list = np.array_split(lon[:, :image_split_range, :], n_image_splits, axis=1)
                                incident_list = np.array_split(incident[:, :image_split_range, :], n_image_splits, axis=1)

                                for i, ride_image in enumerate(ride_image_list):

                                    if gps_flag:
                                        # append lat, lon & incident
                                        ride_image = np.dstack((ride_image, lat_list[i], lon_list[i], incident_list[i]))

                                    else:
                                        # append incident
                                        ride_image = np.dstack((ride_image, incident_list[i]))

                                    if np.any(ride_image[:, :, -1]) > 0:
                                        ride_image[:, :, -1] = 1  # TODO: Maybe preserve incident type
                                        pos_counter += 1

                                        if split == 'train' and data_augmentation_flag:
                                            ride_image_rotated_X = augment_data(ride_image, axis=0)
                                            ride_image_rotated_Y = augment_data(ride_image, axis=1)
                                            ride_image_rotated_Z = augment_data(ride_image, axis=2)
                                            pos_counter += 3

                                            if in_memory_flag:
                                                ride_images_list.append(ride_image_rotated_X)
                                                ride_images_list.append(ride_image_rotated_Y)
                                                ride_images_list.append(ride_image_rotated_Z)

                                            else:
                                                dict_name_rotated_X = os.path.basename(file).replace('.csv', '') +  '_no' + str(i).zfill(5) + '_rotated_X_bucket_incident'
                                                ride_images_dict.update({dict_name_rotated_X: ride_image_rotated_X})
                                                dict_name_rotated_Y = os.path.basename(file).replace('.csv', '') +  '_no' + str(i).zfill(5) + '_rotated_Y_bucket_incident'
                                                ride_images_dict.update({dict_name_rotated_Y: ride_image_rotated_Y})
                                                dict_name_rotated_Z = os.path.basename(file).replace('.csv', '') +  '_no' + str(i).zfill(5) + '_rotated_Z_bucket_incident'
                                                ride_images_dict.update({dict_name_rotated_Z: ride_image_rotated_Z})

                                        if in_memory_flag:
                                            ride_images_list.append(ride_image)

                                        else:
                                            dict_name = os.path.basename(file).replace('.csv', '') + '_no' + str(i).zfill(5) + '_bucket_incident'
                                            ride_images_dict.update({dict_name: ride_image})

                                    else:
                                        ride_image[:, :, -1] = 0
                                        neg_counter += 1
                                        if in_memory_flag:
                                            ride_images_list.append(ride_image)
                                        else:
                                            dict_name = os.path.basename(file).replace('.csv', '') + '_no' + str(i).zfill(5) + '_bucket'
                                            ride_images_dict.update({dict_name: ride_image})
                    except:
                        print(file)

                    class_counts_df[split + '_' + region] = [pos_counter, neg_counter]

                    os.remove(file)

                class_counts_df.to_csv(os.path.join(dir, class_counts_file), ',', index=False)

                os.rmdir(subdir)

                if in_memory_flag:
                    np.savez(os.path.join(dir, split, region + '.npz'), ride_images_list)
                else:
                    np.savez(os.path.join(dir, split, region + '.npz'), **ride_images_dict)

            else:
                with mp.Pool(mp.cpu_count()) as pool:
                    pool.map(partial(create_buckets_inner, bucket_size), file_list)


def fourier_transform_off_memory(dir, split, region, fft_window, slices, gps_flag, imag_flag, file_list):

    ride_data_dict = {}

    data_loaded = np.load(os.path.join(dir, split, region + '.npz'))

    for file in file_list:
        ride_data = data_loaded[file]
        label = ride_data[:, :, -1]

        if gps_flag:
            gps = ride_data[:, :, 6:8]
            ride_data_transformed = np.fft.fft(ride_data[:, :, :-3], axis=0)

            data_transformed_real = np.real(ride_data_transformed)
            data_transformed_imag = np.imag(ride_data_transformed)

            if imag_flag:
                ride_data_transformed = np.concatenate(
                    (data_transformed_real, data_transformed_imag, gps, np.reshape(label, (fft_window, slices, 1))),
                    axis=2)
            else:
                ride_data_transformed = np.concatenate(
                    (data_transformed_real, gps, np.reshape(label, (fft_window, slices, 1))), axis=2)
        else:
            ride_data_transformed = np.fft.fft(ride_data[:, :, :-1], axis=0)

            data_transformed_real = np.real(ride_data_transformed)
            data_transformed_imag = np.imag(ride_data_transformed)

            if imag_flag:
                ride_data_transformed = np.concatenate(
                    (data_transformed_real, data_transformed_imag, np.reshape(label, (fft_window, slices, 1))), axis=2)
            else:
                ride_data_transformed = np.concatenate(
                    (data_transformed_real, np.reshape(label, (fft_window, slices, 1))), axis=2)

        ride_data_dict.update({file: ride_data_transformed})

    return ride_data_dict


def fourier_transform(dir, target_region=None, in_memory_flag=True, deepsense_flag=False, imag_flag=False, gps_flag=False):

    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '*.npz')),
                           desc='apply fourier transform to {} data'.format(split)):
            region = os.path.basename(subdir).replace('.npz','')

            if target_region is not None and target_region != region:
                continue

            ride_data_dict = {}

            if deepsense_flag:

                if in_memory_flag:
                    data_loaded = np.load(os.path.join(dir, split, region + '.npz'))
                    data = data_loaded['arr_0']

                    label = data[:, :, :, -1]

                    if gps_flag:
                        gps = data[:, :, :, 6:8]
                        data_transformed = np.fft.fft(data[:, :, :, :-3], axis=1)
                        data_transformed_real = np.real(data_transformed)
                        data_transformed_imag = np.imag(data_transformed)

                        if imag_flag:
                            data_transformed = np.concatenate((data_transformed_real, data_transformed_imag, gps, np.reshape(label, (-1, fft_window, slices, 1))), axis=3)
                        else:
                            data_transformed = np.concatenate((data_transformed_real, gps, np.reshape(label, (-1, fft_window, slices, 1))), axis=3)

                    else:
                        data_transformed = np.fft.fft(data[:, :, :, :-1], axis=1)
                        data_transformed_real = np.real(data_transformed)
                        data_transformed_imag = np.imag(data_transformed)

                        if imag_flag:
                            data_transformed = np.concatenate((data_transformed_real, data_transformed_imag, np.reshape(label, (-1, fft_window, slices, 1))), axis=3)
                        else:
                            data_transformed = np.concatenate((data_transformed_real, np.reshape(label, (-1, fft_window, slices, 1))), axis=3)

                else:
                    data_loaded = np.load(os.path.join(dir, split, region + '.npz'))
                    file_list_splits = np.array_split(data_loaded.files, len(data_loaded.files))

                    with mp.Pool(mp.cpu_count()) as pool:
                        results = pool.map(partial(fourier_transform_off_memory, dir, split, region, fft_window, slices, gps_flag, imag_flag), file_list_splits)
                        ride_data_dict = {}
                        for result in results:
                            ride_data_dict.update(result)

                if in_memory_flag:
                    np.savez(os.path.join(dir, split, region + '.npz'), data_transformed)
                else:
                    np.savez(os.path.join(dir, split, region + '.npz'), **ride_data_dict)

            else:
                return


def preprocess(dir, target_region=None, bucket_size=100, time_interval=100, interpolation_type='equidistant',
               in_memory_flag=True, deepsense_flag=True, fft_window=5, slices=20, data_augmentation_flag=False, imag_flag=False, gps_flag=False, class_counts_file='class_counts.csv'):
    if gps_flag:
        remove_invalid_rides(dir, target_region)
    remove_acc_outliers(dir, target_region)
    if gps_flag:
        calc_vel_delta(dir, target_region)
    interpolate(dir, target_region, time_interval, interpolation_type)
    if gps_flag:
        remove_vel_outliers(dir, target_region)
    remove_empty_rows(dir, target_region)
    scale(dir, target_region)
    create_buckets(dir, target_region, bucket_size, in_memory_flag, deepsense_flag, fft_window, slices, data_augmentation_flag, gps_flag, class_counts_file)
    fourier_transform(dir, target_region, in_memory_flag, deepsense_flag, imag_flag, gps_flag)

if __name__ == '__main__':
    dir = '../Ride_Data'
    target_region = None
    bucket_size = 100
    time_interval = 100
    interpolation_type = 'equidistant'
    deepsense_flag = True
    fft_window = 5
    slices = 20
    in_memory_flag = True
    data_augmentation_flag = False
    imag_flag = True
    gps_flag = True
    class_counts_file = 'class_counts.csv'
    preprocess(dir, target_region, bucket_size, time_interval, interpolation_type, in_memory_flag, deepsense_flag, fft_window, slices, data_augmentation_flag, imag_flag, gps_flag, class_counts_file)
