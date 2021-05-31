
import os
import glob
from io import StringIO

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import datetime as dt

def export(data_dir, target_dir, info):

    lengths = []
    ride_times = []
    start_time = -1
    incident_counter = 0

    for i, subdir in tqdm(enumerate(os.listdir(data_dir)), desc='loop over regions', total=len(glob.glob(os.path.join(data_dir, '*')))):
        if not subdir.startswith('.'):
            region = subdir

            for j, file in tqdm(enumerate(os.listdir(os.path.join(data_dir, subdir, 'Rides'))), disable=True,
                                desc='loop over rides in {}'.format(region),
                                total=len(glob.glob(os.path.join(os.path.join(data_dir, subdir, 'Rides'), 'VM2_*')))):
                if file.startswith('VM2_'):
                    arr_list = []
                    times = []
                    with open(os.path.join(data_dir, subdir, 'Rides', file)) as f:
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

                                        if int(incident) != -5:
                                            incident_counter += 1
                                            if info:
                                                print('file {} contains an incident'.format(file))

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

                                    if start_time == -1:
                                        start_time = int(line.split(',')[5])
                                    else:
                                        last_time = int(line.split(',')[5])

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

                            ride_time = (last_time - start_time) / 1000 / 60

                            if ride_time <= 180:
                                ride_times.append(ride_time)

                            start_time = -1

                            try:
                                lengths.append(len(arr_list))
                                arr = np.stack(arr_list)

                            except:
                                print('file {} has the wrong format'.format(file))
                                continue

                            s = header + ',bike,childCheckBox,trailerCheckBox,pLoc,incident,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,scary'
                            df = pd.DataFrame(arr, columns=s.split(','))

                            # drop unnecessary columns
                            df.drop(['obsDistanceLeft1', 'obsDistanceLeft2', 'obsDistanceRight1', 'obsDistanceRight2',
                                     'obsClosePassEvent'], 1, inplace=True)

                            # linear interpolation of missing values
                            df.interpolate(inplace=True)

                            if int(file[-1]) in [3, 6, 9]:
                                # test set
                                split = 'test'
                            else:
                                # train set
                                split = 'train'

                            try:
                                os.mkdir(os.path.join(target_dir, 'train'))
                                os.mkdir(os.path.join(target_dir, 'test'))
                            except:
                                pass

                            try:
                                os.mkdir(os.path.join(target_dir, split, region))
                            except:
                                pass

                            df.to_csv(os.path.join(target_dir, split, region, file + '.csv'), ',', index=False)
