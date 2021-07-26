import os
import glob
from io import StringIO
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from fnmatch import fnmatch
import multiprocessing as mp
from functools import partial


def export_file(target_dir, split, file):

    file = file[0]

    region = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))

    arr_list = []
    with open(file) as f:
        system_no = f.readline().partition('#')[0]
        if (system_no.isdecimal() and int(system_no)) >= 73:
            parts = f.read().partition('=========================')
            incident_info = parts[0]
            ride_info = parts[2]
            incident_info_lines = incident_info.splitlines()
            ride_info_lines = ride_info.splitlines()
            incident_info_list = []
            incident_info_tuple = []

            for k, line in enumerate(incident_info_lines):
                if k == 0:
                    incident_header = line
                if k > 0:
                    incident_info_line_split = line.split(',')

                    if incident_info_line_split != '' and incident_info_line_split != ['']:
                        incident_key = incident_info_line_split[0]
                        incident_lat = incident_info_line_split[1]
                        incident_lon = incident_info_line_split[2]
                        incident_ts = incident_info_line_split[3]

                        bike = incident_info_line_split[4]
                        childCheckBox = incident_info_line_split[5]
                        trailerCheckBox = incident_info_line_split[6]
                        pLoc = incident_info_line_split[7]
                        incident = incident_info_line_split[8]
                        i1 = incident_info_line_split[9]
                        i2 = incident_info_line_split[10]
                        i3 = incident_info_line_split[11]
                        i4 = incident_info_line_split[12]
                        i5 = incident_info_line_split[13]
                        i6 = incident_info_line_split[14]
                        i7 = incident_info_line_split[15]
                        i8 = incident_info_line_split[16]
                        i9 = incident_info_line_split[17]
                        i10 = incident_info_line_split[20]
                        scary = incident_info_line_split[18]

                        incident_info_tuple = [incident_key, incident_lat, incident_lon, incident_ts,
                                               bike,
                                               childCheckBox, trailerCheckBox, pLoc, incident, i1, i2,
                                               i3,
                                               i4, i5, i6, i7, i8, i9, i10, scary]

                        incident_info_list.append(incident_info_tuple)

            for k, line in enumerate(ride_info_lines):

                # k == 0 is empty and k == 1 is the system_no
                if k == 2:
                    header = line
                if k > 2:

                    line_arr = None

                    for t in incident_info_list:
                        incident_lat = t[1]
                        incident_lon = t[2]
                        incident_ts = t[3]
                        bike = t[4]
                        childCheckBox = t[5]
                        trailerCheckBox = t[6]
                        pLoc = t[7]
                        incident = t[8]
                        i1 = t[9]
                        i2 = t[10]
                        i3 = t[11]
                        i4 = t[12]
                        i5 = t[13]
                        i6 = t[14]
                        i7 = t[15]
                        i8 = t[16]
                        i9 = t[17]
                        i10 = t[18]
                        scary = t[19]

                        ts = line.split(',')[5]

                        if int(incident) != -5 and (int(ts) == int(incident_ts)):
                            line_arr = np.genfromtxt(
                                StringIO(line + ',' + str(
                                    bike) + ',' + str(childCheckBox) + ',' + str(
                                    trailerCheckBox) + ',' + str(pLoc) + ',' + str(
                                    incident) + ',' + str(i1) + ',' + str(i2) + ',' + str(
                                    i3) + ',' + str(i4) + ',' + str(i5) + ',' + str(i6) + ',' + str(
                                    i7) + ',' + str(i8) + ',' + str(i9) + ',' + str(i10) + ',' + str(
                                    scary)), delimiter=',')

                            # remove t incident from incident_info_list
                            incident_info_list.remove(t)

                    # if there was no fitting incident in the line (incident_ts != ts)
                    if line_arr is None:
                        incident = 0
                        i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, scary = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        line_arr = np.genfromtxt(
                            StringIO(line + ',' + str(
                                bike) + ',' + str(childCheckBox) + ',' + str(
                                trailerCheckBox) + ',' + str(pLoc) + ',' + str(
                                incident) + ',' + str(i1) + ',' + str(i2) + ',' + str(
                                i3) + ',' + str(i4) + ',' + str(i5) + ',' + str(i6) + ',' + str(
                                i7) + ',' + str(i8) + ',' + str(i9) + ',' + str(i10) + ',' + str(
                                scary)), delimiter=',')

                    arr_list.append(line_arr)

            try:
                arr = np.stack(arr_list)
                # check if incident_info_list is empty, print warning if not, as incident will be lost
                if len(incident_info_list) != 0:
                    for t in incident_info_list:
                        if int(t[8]) != -5:
                            print('WARNING: Incident with timestamp ' + t[
                                3] + ' not assigned to a ride_info_line in file ' + file)

            except:
                print('file {} has the wrong format'.format(file))
                return

            s = header + ',bike,childCheckBox,trailerCheckBox,pLoc,incident,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,scary'
            df = pd.DataFrame(arr, columns=s.split(','))

            # drop unnecessary columns
            df.drop(['obsDistanceLeft1', 'obsDistanceLeft2', 'obsDistanceRight1', 'obsDistanceRight2',
                     'obsClosePassEvent'], 1, inplace=True)

            try:
                os.mkdir(os.path.join(target_dir, 'train'))
                os.mkdir(os.path.join(target_dir, 'test'))
                os.mkdir(os.path.join(target_dir, 'val'))
            except:
                pass

            try:
                os.mkdir(os.path.join(target_dir, split, region))
            except:
                pass

            df.to_csv(os.path.join(target_dir, split, region, os.path.basename(file) + '.csv'), ',', index=False)


def export(data_dir, target_dir, target_region=None):
    for subdir in tqdm(glob.glob(os.path.join(data_dir, '[!.]*'))):
        region = os.path.basename(subdir)

        if target_region is not None and target_region != region:
            continue

        file_list = []
        file_names = set()

        root = os.path.join(data_dir, subdir, 'Rides')

        for path, sd, files in os.walk(root):
            for name in files:
                if fnmatch(name, 'VM2_*'):

                    if name not in file_names:
                        file_list.append(os.path.join(path, name))
                        file_names.add(name)

        df = pd.DataFrame(file_list)

        train, val, test = np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])

        with mp.Pool(4) as pool:
            pool.map(partial(export_file, target_dir, 'train'), train.values)
            pool.map(partial(export_file, target_dir, 'val'), val.values)
            pool.map(partial(export_file, target_dir, 'test'), test.values)

        for split in ['train', 'test', 'val']:
            count = 0
            for subdir in tqdm(glob.glob(os.path.join(target_dir, split, '[!.]*')), desc='remove invalid rides in {} data'.format(split)):
                count += len(glob.glob(os.path.join(subdir, 'VM2_*.csv')))
            print('number of rides in {}: {}'.format(split, count))


if __name__ == '__main__':
    data_dir = '../Regions/'
    target_dir = '../Ride_Data/'
    target_region = None
    export(data_dir, target_dir, target_region)
