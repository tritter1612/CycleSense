import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir):
        # super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.num_incidents = 0
        self.files = []
        for path, subdir, files in os.walk(self.data_dir):
            for file in glob(os.path.join(path, '*.csv')):
                if 'incident' in file:
                    self.num_incidents += 1
                self.files.append(file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.genfromtxt(self.files[idx], delimiter=',', skip_header=True)

        y = arr[:, 21]
        x = np.delete(arr, 21, axis=1)  # remove incident label
        x = np.delete(x, 5, axis=1)  # remove timestamp
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y

    def num_labels(self):
        return self.num_incidents


def load_data(data_dir, batch_size=1):
    split = ['train', 'test', 'val']

    train_ds = Dataset(data_dir=os.path.join(data_dir, split[0]))
    val_ds = Dataset(data_dir=os.path.join(data_dir, split[1]))
    test_ds = Dataset(data_dir=os.path.join(data_dir, split[2]))

    train_sample_weights = [0] * len(train_ds)

    for idx, (data, label) in enumerate(train_ds):
        if torch.any(label == 1).item():
            train_sample_weights[idx] = len(train_ds) - train_ds.num_incidents
        else:
            train_sample_weights[idx] = train_ds.num_incidents

    val_sample_weights = [0] * len(val_ds)

    for idx, (data, label) in enumerate(val_ds):
        if torch.any(label == 1).item():
            val_sample_weights[idx] = len(val_ds) - val_ds.num_incidents
        else:
            val_sample_weights[idx] = val_ds.num_incidents

    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)
    val_sampler = WeightedRandomSampler(val_sample_weights, len(val_sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_ds, sampler=train_sampler, num_workers=4, batch_size=batch_size,
                                               prefetch_factor=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, sampler=val_sampler, num_workers=4, batch_size=batch_size,
                                             prefetch_factor=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, num_workers=4, batch_size=batch_size, prefetch_factor=batch_size)

    return train_loader, val_loader, test_loader
