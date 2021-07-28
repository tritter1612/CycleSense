import os
import glob
from fnmatch import fnmatch
import numpy as np
import pandas as pd
import datetime as dt
import multiprocessing as mp
from tqdm.auto import tqdm
from sklearn.preprocessing import MaxAbsScaler
import math
from functools import partial


def remove_unnecessary_cols_inner(file):
    df = pd.read_csv(file)

    # drop columns which are no longer needed
    df.drop(columns=['bike', 'childCheckBox', 'trailerCheckBox', 'pLoc', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',
                     'i9', 'i10', 'scary'], inplace=True)

    df.to_csv(file, ',', index=False)


def remove_unnecessary_cols(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='remove unnecessary cols in {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))

            with mp.Pool(4) as pool:
                pool.map(remove_unnecessary_cols_inner, file_list)


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
                    # remove rides where one col is completely empty or timestamp interval is too long
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
        df.loc[outlier_rows, 'acc'] = ''

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

            with mp.Pool(4) as pool:
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


def rotateRides_inner(file):
    arr = np.genfromtxt(file, delimiter=',', skip_header=True)

    for i in range(arr.shape[0]):
        R = getRotationMatrixFromVector(arr[i, 13], arr[i, 14], arr[i, 15], arr[i, 16])
        XYZ = np.matmul(R.T, np.array([arr[i, 2], arr[i, 3], arr[i, 4]]))
        abc = np.matmul(R.T, np.array([arr[i, 7], arr[i, 8], arr[i, 9]]))
        XLYLZL = np.matmul(R.T, np.array([arr[i, 10], arr[i, 11], arr[i, 12]]))

        arr[i, [[2, 3, 4, 7, 8, 9, 10, 11, 12]]] = [XYZ[0], XYZ[1], XYZ[2], abc[0], abc[1], abc[2], XLYLZL[0],
                                                    XLYLZL[1], XLYLZL[2]]

    np.savetxt(file, arr, delimiter=',',
               header='lat,lon,X,Y,Z,timeStamp,acc,a,b,c,XL,YL,ZL,RX,RY,RZ,RC,incident',
               comments='')


def rotateRides(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='rotate rides in {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))

            with mp.Pool(4) as pool:
                pool.map(rotateRides_inner, file_list)


def linear_interpolate(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='apply linear interpolation on {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):
                df = pd.read_csv(file)

                # convert timestamp to datetime format
                df['timeStamp'] = df['timeStamp'].apply(
                    lambda x: dt.datetime.utcfromtimestamp(x / 1000))

                # set timeStamp col as pandas datetime index
                df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='ns')

                df = df.set_index(pd.DatetimeIndex(df['timeStamp']))

                # drop all duplicate occurrences of the labels and keep the first occurrence
                df = df[~df.index.duplicated(keep='first')]

                # interpolation of acc via linear interpolation based on timestamp
                df['acc'].interpolate(method='time', inplace=True)

                df.sort_index(axis=0, ascending=False, inplace=True)

                # interpolation of missing values via padding on the reversed df
                df['lat'].interpolate(method='pad', inplace=True)
                df['lon'].interpolate(method='pad', inplace=True)

                df.sort_index(axis=0, ascending=True, inplace=True)

                df['timeStamp'] = df.index.astype(np.int64)

                df.to_csv(file, ',', index=False)


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
                df.dropna(inplace=True, axis=0)

                if len(df) != 0:
                    df.to_csv(file, ',', index=False)
                else:
                    os.remove(file)


def getRotationMatrixFromVector(RX, RY, RZ, RC):
    R = np.zeros((3, 3))

    sq_q1 = 2 * RX * RX
    sq_q2 = 2 * RY * RY
    sq_q3 = 2 * RZ * RZ
    q1_q2 = 2 * RX * RY
    q3_q0 = 2 * RZ * RC
    q1_q3 = 2 * RX * RZ
    q2_q0 = 2 * RY * RC
    q2_q3 = 2 * RY * RZ
    q1_q0 = 2 * RX * RC

    R[0, 0] = 1 - sq_q2 - sq_q3
    R[1, 0] = q1_q2 - q3_q0
    R[2, 0] = q1_q3 + q2_q0
    R[0, 1] = q1_q2 + q3_q0
    R[1, 1] = 1 - sq_q1 - sq_q3
    R[2, 1] = q2_q3 - q1_q0
    R[0, 2] = q1_q3 - q2_q0
    R[1, 2] = q2_q3 + q1_q0
    R[2, 2] = 1 - sq_q1 - sq_q2

    return R


def scale(dir, target_region=None):
    scaler_maxabs = MaxAbsScaler()

    split = 'train'

    for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                       desc='fit scaler'):
        region = os.path.basename(subdir)

        if target_region is not None and target_region != region:
            continue

        for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):
            df = pd.read_csv(file)

            df.fillna(0, inplace=True)

            scaler_maxabs.partial_fit(df[['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c', 'XL', 'YL', 'ZL']])

    print(scaler_maxabs.max_abs_)

    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')), desc='scale {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            for file in glob.glob(os.path.join(subdir, 'VM2_*.csv')):
                df = pd.read_csv(file)
                df[['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c', 'XL', 'YL',
                    'ZL']] = scaler_maxabs.transform(
                    df[['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c', 'XL', 'YL', 'ZL']])

                df.to_csv(file, ',', index=False)


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
                df_split.to_csv(file.replace('.csv', '') + '_no' + str(k) + '_bucket_incident.csv', ',', index=False)
            else:
                df_split['incident'] = 0.0
                df_split.to_csv(file.replace('.csv', '') + '_no' + str(k) + '_bucket.csv', ',', index=False)

    os.remove(file)


def create_buckets(dir, target_region=None, bucket_size=22):
    for split in ['train', 'test', 'val']:

        for subdir in tqdm(glob.glob(os.path.join(dir, split, '[!.]*')),
                           desc='generate buckets for {} data'.format(split)):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))

            with mp.Pool(4) as pool:
                pool.map(partial(create_buckets_inner, bucket_size), file_list)


def preprocess(dir, target_region=None, bucket_size=22):
    remove_unnecessary_cols(dir, target_region)
    remove_invalid_rides(dir, target_region)
    remove_acc_outliers(dir, target_region)
    calc_vel_delta(dir, target_region)
    rotateRides(dir, target_region)
    linear_interpolate(dir, target_region)
    remove_vel_outliers(dir, target_region)
    remove_empty_rows(dir, target_region)
    scale(dir, target_region)
    create_buckets(dir, target_region, bucket_size)


if __name__ == '__main__':
    dir = '../Ride_Data'
    target_region = None
    bucket_size = 22
    preprocess(dir, target_region, bucket_size)
