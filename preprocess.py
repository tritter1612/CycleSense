import os
import glob

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MaxAbsScaler
import datetime as dt


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


def calc_gps_delta(dir, target_region=None):
    for split in ['train', 'test', 'val']:

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))),
                              desc='calculate gps delta on {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                if target_region is not None and target_region != region:
                    continue

                for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                    desc='loop over rides in {}'.format(region),
                                    total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):
                    df = pd.read_csv(os.path.join(dir, split, subdir, file))

                    df[['lat', 'lon']] = df[['lat', 'lon']].diff().fillna(0)

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
    linear_interpolate(dir, target_region)
    calc_gps_delta(dir, target_region)
    scale(dir, target_region)
