import os
import glob
import numpy as np


def get_metrics(dir, target_region=None, bucket_size=10000):
    for split in ['test']:
        tn, fp, fn, tp = 0, 0, 0, 0

        for subdir in glob.glob(os.path.join(dir, split, '[!.]*')):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))

            for file in file_list:
                arr = np.genfromtxt(file, delimiter=',', skip_header=True)
                if len(arr.shape) < 2:
                    # ride too short
                    continue
                incs = arr[:, -1]
                timestamp = arr[:, 5]
                # take only XYZ columns
                arr = arr[:, 2:5]

                start = timestamp[0]
                end = timestamp[-1]
                n_window_splits = int((end - start) // bucket_size)
                buckets = np.zeros((n_window_splits, 3))
                incidents = np.zeros(n_window_splits)
                heuristic_findings = np.zeros(n_window_splits)
                upper_ts = start
                j = 0
                if n_window_splits == 0:
                    continue
                for i in range(n_window_splits):
                    lower_ts = upper_ts
                    upper_ts = lower_ts + bucket_size  # ms
                    if upper_ts > end:
                        upper_ts = end
                    lower = j
                    while j < timestamp.shape[0] and timestamp[j] < upper_ts:
                        j += 1
                    upper = j
                    if upper > lower:
                        actual_bucket = arr[lower:upper, :]
                        max = np.nanmax(actual_bucket, axis=0)
                        min = np.nanmin(actual_bucket, axis=0)
                        buckets[i, :] = max - min
                        if np.any(incs[lower:upper]) > 0:
                            incidents[i] = 1.0

                # find the 2 argmax (inner loop) for each column (outer loop)
                # duplicates are not tolerated and are replaced -> exactly 6 findings per ride
                for k in range(3):
                    for i in range(2):
                        argmax = np.argmax(buckets[:, k], axis=0)
                        heuristic_findings[argmax] = 1.0
                        # set entry to -1.0 for whole bucket to allow argmax to find the next biggest entries
                        buckets[argmax, :] = -1.0

                # match incidents against findings to compute confusion matrix
                for i in range(n_window_splits):
                    if heuristic_findings[i] > 0.0:
                        if incidents[i] > 0.0:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if incidents[i] > 0.0:
                            fn += 1
                        else:
                            tn += 1
        print('Confusion Matrix for {} dataset:'.format(split))
        print(tn, fp)
        print(fn, tp)
        print('Specificity: ' + str(1 - fp / (fp + tn)))
        print('Sensitivity: ' + str(tp / (fn + tp)))

        return tn, fp, fn, tp