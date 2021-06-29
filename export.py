import os
import glob
from io import StringIO
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from fnmatch import fnmatch
import multiprocessing as mp
from functools import partial


def export_file(target_dir, file):
    region = os.path.basename(os.path.dirname(os.path.dirname(file)))

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
                if k == 2:
                    header = line
                if k > 2:

                    device_os = 1 if 'i' in system_no else 0

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

                        else:
                            incident = 0 if incident != -5 else -5

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

            except:
                print('file {} has the wrong format'.format(file))
                return

            s = header + ',bike,childCheckBox,trailerCheckBox,pLoc,incident,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,scary'
            df = pd.DataFrame(arr, columns=s.split(','))

            # drop unnecessary columns
            df.drop(['obsDistanceLeft1', 'obsDistanceLeft2', 'obsDistanceRight1', 'obsDistanceRight2',
                     'obsClosePassEvent'], 1, inplace=True)

            if int(file[-1]) in [6, 7]:
                # test set
                split = 'test'
            elif int(file[-1]) in [8, 9]:
                # validation set
                split = 'val'
            else:
                # train set
                split = 'train'

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
    for i, subdir in tqdm(enumerate(os.listdir(data_dir)), desc='loop over regions',
                          total=len(glob.glob(os.path.join(data_dir, '*')))):
        if not subdir.startswith('.'):
            region = subdir

            if target_region is not None and target_region != region:
                continue

            file_list = []

            root = os.path.join(data_dir, subdir, 'Rides')

            for path, sd, files in os.walk(root):
                for name in files:
                    if fnmatch(name, 'VM2_*'):
                        file_list.append(os.path.join(path, name))

            with mp.Pool(4) as pool:
                pool.map(partial(export_file, target_dir), file_list)
