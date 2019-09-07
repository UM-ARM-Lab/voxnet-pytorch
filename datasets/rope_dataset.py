import torch
from torch.utils.data import Dataset

import numpy as np
import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class RopeDataset(Dataset):

    def __init__(self, data_dir, samples_per_file):
        super(RopeDataset, self).__init__()

        self.n_per_file = samples_per_file
        self.files = os.listdir(data_dir)
        self.data_dir = data_dir
        # Files should have number as suffix
        self.files.sort(key=lambda x: int(re.split("_|\.", x)[-2]))
        self.start_pos = int(re.split("_|\.", self.files[0])[1])

    def __len__(self):
        return len(self.files) * self.n_per_file

    def __getitem__(self, idx):
        # Need to find out which file contains item
        if isinstance(idx, slice):
            raise ValueError("Slice operation not currently supported")

        file_containing_item = self.data_dir + '/' + self.files[idx // self.n_per_file]
        data = np.load(file_containing_item, mmap_mode='r')

        key = str(idx + self.start_pos)

        # Load the stuff
        local_env = data[key + '/local_env']
        band_pre = data[key + '/band_pre']
        band_post = data[key + '/band_post']
        label = data[key + '/label']

        x = np.stack((local_env, band_pre, band_post), axis=0)
        return x, label


if __name__ == '__main__':

    rope_dataset = RopeDataset('../data/rope/train', samples_per_file=1024)
    print(len(rope_dataset))
    rope_dataset[0]


