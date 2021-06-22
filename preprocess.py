import os
import glob
from fnmatch import fnmatch
import numpy as np
import pandas as pd
import datetime as dt
import multiprocessing as mp
from tqdm.auto import tqdm
from sklearn.preprocessing import MaxAbsScaler
from functools import partial


def remove_invalid_rides(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))),
                              desc='remove invalid rides in {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                if target_region is not None and target_region != region:
                    continue

                for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                    desc='loop over rides in {}'.format(region),
                                    total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):

                    if file.startswith('VM2_'):

                        df = pd.read_csv(os.path.join(dir, split, subdir, file))

                        df_cp = df.copy(deep=True)
                        df_cp['timeStamp'] = df_cp['timeStamp'].diff()

                        breakpoints = np.where((df_cp['timeStamp'] > 6000).to_numpy())

                        if len(breakpoints[0]) > 0:
                            os.remove(os.path.join(dir, split, subdir, file))


def replace_outlier_file(lower, upper, file):
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

    for i, subdir in enumerate(os.listdir(os.path.join(dir, split))):
        if not subdir.startswith('.'):
            region = subdir

            if target_region is not None and target_region != region:
                continue

            for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                desc='loop over rides in {}'.format(region),
                                total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):
                df = pd.read_csv(os.path.join(dir, split, subdir, file))

                df = df.dropna()

                if df.shape[0] == 0:
                    os.remove(os.path.join(dir, split, subdir, file))

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

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))),
                              desc='remove accuracy outliers from {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                if target_region is not None and target_region != region:
                    continue

                file_list = []

                root = os.path.join(dir, split, subdir)

                for path, sd, files in os.walk(root):
                    for name in files:
                        if fnmatch(name, 'VM2_*.csv'):
                            file_list.append(os.path.join(path, name))

                with mp.Pool(4) as pool:
                    pool.map(partial(replace_outlier_file, lower, upper), file_list)


def calc_vel_delta(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))),
                              desc='calculate vel delta on {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                if target_region is not None and target_region != region:
                    continue

                for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                    desc='loop over rides in {}'.format(region),
                                    total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):

                    if file.startswith('VM2_'):
                        df = pd.read_csv(os.path.join(dir, split, subdir, file))

                        df_cp = df.copy(deep=True)

                        df_cp = df_cp.dropna()
                        df_cp[['lat', 'lon', 'timeStamp']] = df_cp[['lat', 'lon', 'timeStamp']].diff()
                        df_cp = df_cp.dropna()

                        # compute lat & lon change per second
                        df_cp['lat'] = df_cp['lat'] * 1000 / df_cp['timeStamp']
                        df_cp['lon'] = df_cp['lon'] * 1000 / df_cp['timeStamp']

                        df[['lat', 'lon']] = df_cp[['lat', 'lon']]

                        df.to_csv(os.path.join(dir, split, subdir, file), ',', index=False)


def linear_interpolate(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))),
                              desc='apply linear interpolation on {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                if target_region is not None and target_region != region:
                    continue

                for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                    desc='loop over rides in {}'.format(region),
                                    total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):

                    if file.startswith('VM2_'):
                        df = pd.read_csv(os.path.join(dir, split, subdir, file))

                        # convert timestamp to datetime format
                        df['timeStamp'] = df['timeStamp'].apply(
                            lambda x: dt.datetime.utcfromtimestamp(x / 1000).isoformat())

                        # set timeStamp col as pandas datetime index
                        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
                        df = df.set_index(pd.DatetimeIndex(df['timeStamp']))

                        # interpolation of acc via linear interpolation based on timestamp
                        df['acc'].interpolate(method='time', inplace=True)

                        df.sort_index(axis=0, ascending=False, inplace=True)

                        # interpolation of missing values via padding on the reversed df
                        df['lat'].interpolate(method='pad', inplace=True)
                        df['lon'].interpolate(method='pad', inplace=True)

                        df.sort_index(axis=0, ascending=True, inplace=True)

                        df.to_csv(os.path.join(dir, split, subdir, file), ',', index=False)


def remove_vel_outliers(dir, target_region=None):
    l = []
    split = 'train'

    for i, subdir in enumerate(os.listdir(os.path.join(dir, split))):
        if not subdir.startswith('.'):
            region = subdir

            if target_region is not None and target_region != region:
                continue

            for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                desc='loop over rides in {}'.format(region),
                                total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):

                if file.startswith('VM2_'):

                    df = pd.read_csv(os.path.join(dir, split, subdir, file))

                    df = df.dropna()

                    if df.shape[0] == 0:
                        os.remove(os.path.join(dir, split, subdir, file))

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

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))),
                              desc='remove velocity outliers from {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                if target_region is not None and target_region != region:
                    continue

                for j, file in enumerate(os.listdir(os.path.join(dir, split, subdir))):

                    if file.startswith('VM2_'):
                        df = pd.read_csv(os.path.join(dir, split, subdir, file))

                        arr = df[['lat', 'lon']].to_numpy()

                        outliers_lower = arr < lower
                        outliers_upper = arr > upper

                        outliers = np.logical_or(outliers_lower, outliers_upper)
                        outliers_bool = np.any(outliers, axis=1)
                        outlier_rows = np.where(outliers_bool)[0]

                        if len(outlier_rows) > 0:
                            df = df.drop(outlier_rows)
                            df.to_csv(os.path.join(dir, split, subdir, file), ',', index=False)


def scale(dir, target_region=None):
    scaler_maxabs = MaxAbsScaler()

    split = 'train'

    for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))), desc='fit scaler',
                          total=len(glob.glob(os.path.join(dir, split, '*')))):
        if not subdir.startswith('.'):
            region = subdir

            if target_region is not None and target_region != region:
                continue

            for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                desc='loop over rides in {}'.format(region),
                                total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):

                if file.startswith('VM2_'):
                    df = pd.read_csv(os.path.join(dir, split, subdir, file))

                    df.fillna(0, inplace=True)

                    scaler_maxabs.partial_fit(df[['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c', 'XL', 'YL', 'ZL']])

    for split in ['train', 'test', 'val']:

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))), desc='scale {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                if target_region is not None and target_region != region:
                    continue

                for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                    desc='loop over rides in {}'.format(region),
                                    total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):

                    if file.startswith('VM2_'):
                        df = pd.read_csv(os.path.join(dir, split, subdir, file))
                        df[['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c', 'XL', 'YL',
                            'ZL']] = scaler_maxabs.transform(
                            df[['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c', 'XL', 'YL', 'ZL']])

                        df.to_csv(os.path.join(dir, split, subdir, file), ',', index=False)


def preprocess(dir, target_region=None):
    remove_invalid_rides(dir, target_region)
    remove_acc_outliers(dir, target_region)
    calc_vel_delta(dir, target_region)
    linear_interpolate(dir, target_region)
    remove_vel_outliers(dir, target_region)
    scale(dir, target_region)
