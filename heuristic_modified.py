import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix


# extend heuristic for buckets to measure AUC ROC:
# for each bucket the acceleration difference on all 3 axes is calculated
# and then the max of these 3 numbers is assigned as score to the bucket
def get_metrics(dir, target_region=None, bucket_size=10000):
    for split in ['test']:
        y = []
        scores = []

        for subdir in glob.glob(os.path.join(dir, split, '[!.]*')):
            region = os.path.basename(subdir)

            if target_region is not None and target_region != region:
                continue

            file_list = glob.glob(os.path.join(subdir, 'VM2_*.csv'))
            num_rides = 0

            for file in file_list:
                arr = np.genfromtxt(file, delimiter=',', skip_header=True)
                if len(arr.shape) < 2:
                    # ride too short
                    continue
                # maintained incidents
                num_rides += 1
                incs = arr[:, -1]
                timestamp = arr[:, 5]
                # take only XYZ columns
                arr = arr[:, 2:5]

                start = timestamp[0]
                end = timestamp[-1]
                n_window_splits = int((end - start) // bucket_size)
                buckets = np.zeros((n_window_splits, 3))
                buckets_max = np.zeros(n_window_splits)
                incidents = np.zeros(n_window_splits)
                upper_ts = start
                j = 0  # counter for the entries in a ride
                if n_window_splits == 0:
                    continue

                # extend heuristic for buckets to measure AUC ROC:
                # for each bucket the acceleration difference on all 3 axes is calculated
                # and then the max of these 3 numbers is assigned as score to the bucket
                for i in range(n_window_splits):
                    # go thru bucket i
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
                        # store max diff of the 3 axes in buckets_max as heuristical score
                        buckets_max[i] = np.max(buckets[i, :])
                        if np.any(incs[lower:upper]) > 0:
                            # ith bucket contains an incidents
                            incidents[i] = 1
                    else:
                        # bucket has only one entry
                        if incs[lower] > 0:
                            incidents[i] = 1
                        buckets_max[i] = 0.0

                # add ground truth incidents and heuristic scores for this ride to overall y and scores
                y = np.concatenate((y, incidents))
                scores = np.concatenate((scores, buckets_max))

        num_buckets = y.shape[0]

        roc_auc = roc_auc_score(y, scores)
        print(split + ' AUC ROC: ' + str(roc_auc) + ' for ' + str(num_buckets) + ' buckets of ' + str(bucket_size) + 'ms')
        fpr, tpr, threshold = roc_curve(y, scores)

        # 6 proposals per ride at average
        print("Rides: " + str(num_rides))
        fpr_allowed = 6 * num_rides / num_buckets
        print("Specificity for at average 6 proposals per ride has to be at least: " + str(1 - fpr_allowed))
        # find the last fpr which is <= the allowed fpr by going thru the ordered fpr list
        for i in range(fpr.shape[0]):
            if fpr[i] <= fpr_allowed:
                # print(1 - fpr[i], tpr[i], threshold[i])
                last = i
        # print sensitivity at specificity
        print("Sensitivity at This specificity with Threshold:")
        print(tpr[last], 1 - fpr[last], threshold[last])
        tn, fp, fn, tp = confusion_matrix(y, scores >= threshold[last]).ravel()
        print("Confusion Matrix:")
        print(tn, fp)
        print(fn, tp)

        return tn, fp, fn, tp, fpr, tpr, roc_auc
