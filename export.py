import os
import glob
from io import StringIO
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from fnmatch import fnmatch
import multiprocessing as mp
from functools import partial


def export_file(target_dir, region, split, lin_acc_flag, file):

    file = file[0]

    arr_list = []
    with open(file) as f:
        system = None
        system_no = f.readline().partition('#')[0]
        if system_no.isdecimal():
            parts = f.read().partition('=========================')
            system = 'a'
        elif system_no.startswith('i'):
            parts = f.read().partition('===================')
            system = 'i'
            system_no = system_no[1:]

        if (system == 'a' and system_no.isdecimal() and int(system_no) >= 48) or (system == 'i' and int(system_no) >= 5 and int(system_no) <= 15 and int(system_no) != 14 and int(system_no) != 6):

            system = 'o' if (system == 'a' and int(system_no) < 73) else system

            incident_info = parts[0]
            ride_info = parts[2]
            incident_info_lines = incident_info.splitlines()
            ride_info_lines = ride_info.splitlines()
            incident_info_list = []

            if len(ride_info_lines) < 5:
                print('file {} has the wrong format'.format(file))
                return

            start = int(ride_info_lines[3].split(',')[5])
            end = int(ride_info_lines[-1].split(',')[5])

            for k, ride_line in enumerate(incident_info_lines):
                if k == 0:
                    incident_header = ride_line
                if k > 0:
                    incident_info_line_split = ride_line.split(',')

                    if incident_info_line_split != '' and incident_info_line_split != ['']:
                        incident_key = incident_info_line_split[0]
                        incident_lat = incident_info_line_split[1]
                        incident_lon = incident_info_line_split[2]
                        incident_ts = incident_info_line_split[3]
                        incident = incident_info_line_split[8]

                        incident_info_tuple = [incident_key, incident_lat, incident_lon, incident_ts, incident]

                        if incident_ts != '' and incident_ts != '1337' and (int(incident_ts) < int(start) or int(incident_ts) > int(end)):
                            print('WARNING: Incident with timestamp ' + incident_ts + ' does not occur during ride ' + file)
                            return

                        found = False
                        if len(incident_info_list) > 0:
                            for info_tuple in incident_info_list:
                                if incident_ts == info_tuple[3] and incident_ts != '1337':
                                    # incident_ts already in the list
                                    found = True
                        if not found:
                            # new incident: add it to list
                            incident_info_list.append(incident_info_tuple)

            for k, ride_line in enumerate(ride_info_lines):

                # k == 0 is empty and k == 1 is the system_no
                if k == 2:
                    # lat = line_split[0]
                    # lon = line_split[1]
                    # X = line_split[2]
                    # Y = line_split[3]
                    # Z = line_split[4]
                    # ts = line_split[5]
                    # acc = line_split[6]
                    # a = line_split[7]
                    # b = line_split[8]
                    # c = line_split[9]
                    header_split = ride_line.split(',')
                    header = header_split[:10]
                    if (system == 'a') and lin_acc_flag:
                        # newer android data have more data columns:
                        # XL = line_split[15]
                        # YL = line_split[16]
                        # ZL = line_split[17]
                        # RX = line_split[18]
                        # RY = line_split[19]
                        # RZ = line_split[20]
                        # RC = line_split[21]
                        header = header + header_split[15:18]
                    header = ','.join(header)

                if k > 2:
                    ride_line_arr = None
                    line_split = ride_line.split(',')
                    lat = line_split[0]
                    lon = line_split[1]
                    ts = line_split[5]

                    ride_line_data = line_split[:10]
                    if (system == 'a') and lin_acc_flag:
                        # for new android data more columns are exported
                        ride_line_data = ride_line_data + line_split[15:18]

                    for t in incident_info_list:
                        # f has to be a copy of fields, to avoid that more than one incident is
                        # appended to one ride line. The last one wins.
                        f = ride_line_data.copy()
                        incident_key = t[0]
                        incident_lat = t[1]
                        incident_lon = t[2]
                        incident_ts = t[3]
                        incident = t[4]

                        if ts != '' and incident_ts != '':

                            # check if incident timestamp belongs to a timestamp in the ride data;
                            # else check if manually added incident gps coordinates match any lat and lon data
                            # incident_ts == 1337 --> manually added incident (21 <= android system_no <= 71)
                            if (int(incident) != -5 and ts == incident_ts) or (incident_ts == '1337' and lat == incident_lat and lon == incident_lon):
                                f.append(incident)
                                ride_line_arr = np.genfromtxt(StringIO(','.join(f)), delimiter=',')

                                # remove t incident from incident_info_list
                                incident_info_list.remove(t)

                            # check if incident_ts is already after ts
                            # then the incident is assigned to this ride line
                            # unless timpstamp is '1337'
                            elif int(incident) != -5 and int(ts) > int(incident_ts) and incident_ts != '1337':
                                f.append(incident)
                                ride_line_arr = np.genfromtxt(StringIO(','.join(f)), delimiter=',')

                                # remove t incident from incident_info_list
                                incident_info_list.remove(t)

                    # if there was no fitting incident in the line (incident_ts != ts)
                    if ride_line_arr is None:
                        incident = '0'
                        ride_line_data.append(incident)
                        ride_line_arr = np.genfromtxt(StringIO(','.join(ride_line_data)), delimiter=',')

                    arr_list.append(ride_line_arr)

            # check if all incidents could be assigned properly, if not remove ride
            for t in incident_info_list:
                if int(t[4]) != -5:
                    print('WARNING: Incident with timestamp ' + t[3] + ' not assigned to a ride_info_line in file ' + file)
                    return

            try:
                arr = np.stack(arr_list)

            except:
                print('file {} has the wrong format'.format(file))
                return

            # if a, b, c are all 0.0
            if np.all(arr[:,7:10] == 0.0):
                print('file {} has the wrong format, abc are all 0.0'.format(file))
                return

            if (system == 'a') and lin_acc_flag:
                if np.all(arr[:,10:13] == 0.0):
                    print('file {} has the wrong format, linear accelerometer are all 0.0'.format(file))
                    return

            s = header + ',' + 'incident'
            df = pd.DataFrame(arr, columns=s.split(','))

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

            df.to_csv(os.path.join(target_dir, split, region, os.path.basename(file) + system + '.csv'), ',', index=False)


def export(data_dir, target_dir, target_region=None, lin_acc_flag=False):
    for subdir in tqdm(glob.glob(os.path.join(data_dir, '[!.]*'))):
        region = os.path.basename(subdir)

        if target_region is not None and target_region != region:
            continue

        file_list = []
        file_names = set()

        root = os.path.join(data_dir, region, 'Rides')

        for path, sd, files in os.walk(root):
            for name in files:
                if fnmatch(name, 'VM2_*'):

                    if name not in file_names:
                        file_list.append(os.path.join(path, name))
                        file_names.add(name)

        df = pd.DataFrame(file_list)

        train, val, test = np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])

        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(partial(export_file, target_dir, target_region, 'train', lin_acc_flag), train.values)
            pool.map(partial(export_file, target_dir, target_region, 'val', lin_acc_flag), val.values)
            pool.map(partial(export_file, target_dir, target_region, 'test', lin_acc_flag), test.values)

    for split in ['train', 'test', 'val']:
        count = 0
        for subdir in tqdm(glob.glob(os.path.join(target_dir, split, '[!.]*')), desc='count rides in {} data'.format(split)):
            count += len(glob.glob(os.path.join(subdir, 'VM2_*.csv')))
        print('number of rides in {}: {}'.format(split, count))


if __name__ == '__main__':
    data_dir = '../Regions/'
    target_dir = '../Ride_Data/'
    target_region = 'Berlin'
    lin_acc_flag = False
    export(data_dir, target_dir, target_region, lin_acc_flag)
