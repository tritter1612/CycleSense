import os
import glob

import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import warnings

warnings.filterwarnings("ignore")


def linear_interpolate(dir):
    for split in ['train', 'test']:

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))),
                              desc='apply linear interpolation on {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                    desc='loop over rides in {}'.format(region),
                                    total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):
                    df = pd.read_csv(os.path.join(dir, split, subdir, file))

                    # set timeStamp col as pandas datetime index
                    df['timeStamp'] = df['timeStamp'].apply(
                        lambda x: dt.datetime.utcfromtimestamp(x / 1000).strftime('%d.%m.%Y %H:%M:%S,%f'))
                    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
                    df = df.set_index(pd.DatetimeIndex(df['timeStamp']))

                    # linear interpolation of missing values
                    df['lat'].interpolate(method='time', inplace=True)
                    df['lon'].interpolate(method='time', inplace=True)
                    df['acc'].interpolate(method='time', inplace=True)

                    df.to_csv(os.path.join(dir, split, subdir, file), ',', index=False)


def calc_gps_delta(dir):
    for split in ['train', 'test']:

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))),
                              desc='apply linear interpolation on {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                    desc='loop over rides in {}'.format(region),
                                    total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):
                    df = pd.read_csv(os.path.join(dir, split, subdir, file))

                    df[['lat', 'lon']] = df[['lat', 'lon']].diff().fillna(0)

                    df.to_csv(os.path.join(dir, split, subdir, file), ',', index=False)


def scale(dir):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    acc_scaler = MinMaxScaler(feature_range=(0, 1))

    split = 'train'

    for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))), desc='fit scaler',
                          total=len(glob.glob(os.path.join(dir, split, '*')))):
        if not subdir.startswith('.'):
            region = subdir

            for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                desc='loop over rides in {}'.format(region),
                                total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):

                df = pd.read_csv(os.path.join(dir, split, subdir, file))

                if df[['XL', 'YL', 'ZL']].isnull().values.any():
                    os.remove(os.path.join(dir, split, subdir, file))

                scaler.partial_fit(df[['X', 'Y', 'Z', 'a', 'b', 'c', 'XL', 'YL', 'ZL']])
                acc_scaler.partial_fit(df[['acc']])

    for split in ['train', 'test']:

        for i, subdir in tqdm(enumerate(os.listdir(os.path.join(dir, split))), desc='scale {} data'.format(split),
                              total=len(glob.glob(os.path.join(dir, split, '*')))):
            if not subdir.startswith('.'):
                region = subdir

                for j, file in tqdm(enumerate(os.listdir(os.path.join(dir, split, subdir))), disable=True,
                                    desc='loop over rides in {}'.format(region),
                                    total=len(glob.glob(os.path.join(dir, split, subdir, 'VM2_*')))):
                    df = pd.read_csv(os.path.join(dir, split, subdir, file))
                    df[['X', 'Y', 'Z', 'a', 'b', 'c', 'XL', 'YL', 'ZL']] = scaler.transform(
                        df[['X', 'Y', 'Z', 'a', 'b', 'c', 'XL', 'YL', 'ZL']])
                    df[['acc']] = acc_scaler.transform(df[['acc']])

                    df.to_csv(os.path.join(dir, split, subdir, file), ',', index=False)


def preprocess(dir):
    linear_interpolate(dir)
    calc_gps_delta(dir)
    scale(dir)
