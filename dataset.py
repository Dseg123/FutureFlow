import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class TimeSeriesDataset(Dataset):

    def __init__(self, csv_file, column_names, x_len, y_len=1, diff_order = 0, train = True, frac=0.8, rand_seed = 100, size=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.series_df = pd.read_csv(csv_file)[column_names]
        if size:
            self.series_df = self.series_df.iloc[:size]
            
        self.x_len = x_len
        self.y_len = y_len
        
        self.diff_order = diff_order
    
        if train:
            self.indices = np.arange(0, int(frac * len(self.series_df)) - self.x_len - self.y_len - self.diff_order)
        else:
            self.indices = np.arange(int(frac * len(self.series_df)), len(self.series_df) - self.x_len - self.y_len - self.diff_order)
        
        self.frac = frac
        self.train = train
        np.random.seed(rand_seed)
        np.random.shuffle(self.indices)


    def __len__(self):
        return len(self.indices)
        # ratio = self.frac
        # if not self.train:
        #     ratio = 1 - ratio

        # return int(len(self.indices) * ratio)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # if not self.train:
        #     idx += int(len(self.indices) * self.frac)
        
        idx = self.indices[idx]
        
        sample = self.series_df.iloc[idx:idx + self.x_len + self.y_len + self.diff_order]
        for i in range(self.diff_order):
            sample = sample - sample.shift(1)
            sample = sample.iloc[1:]
        sample = sample.reset_index().drop(columns=["index"])

        sample_X = sample.iloc[:self.x_len].to_numpy()
        sample_y = sample.iloc[self.x_len:].to_numpy()

        X = torch.tensor(sample_X, dtype=torch.float32)
        y = torch.tensor(sample_y, dtype=torch.float32)
        return X, y

class TimeSeriesAvgDataset(Dataset):

    def __init__(self, csv_file, column_names, x_len, y_len=1, diff_order = 0, train = True, frac=0.8, rand_seed = 100):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.series_df = pd.read_csv(csv_file)[column_names]
        self.x_len = x_len
        self.y_len = y_len
        
        self.diff_order = diff_order
    
        if train:
            self.indices = np.arange(0, int(frac * len(self.series_df)) - self.x_len - self.y_len - self.diff_order)
        else:
            self.indices = np.arange(int(frac * len(self.series_df)), len(self.series_df) - self.x_len - self.y_len - self.diff_order)
        
        self.frac = frac
        self.train = train
        np.random.seed(rand_seed)
        np.random.shuffle(self.indices)


    def __len__(self):
        return len(self.indices)
        # ratio = self.frac
        # if not self.train:
        #     ratio = 1 - ratio

        # return int(len(self.indices) * ratio)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # if not self.train:
        #     idx += int(len(self.indices) * self.frac)
        
        idx = self.indices[idx]
        
        sample = self.series_df.iloc[idx:idx + self.x_len + self.y_len + self.diff_order]
        for i in range(self.diff_order):
            sample = sample - sample.shift(1)
            sample = sample.iloc[1:]
        sample = sample.reset_index().drop(columns=["index"])

        sample_X = sample.iloc[:self.x_len].to_numpy()
        sample_y = sample.iloc[self.x_len:].to_numpy()
        sample_y = np.mean(sample_y, axis=0)

        X = torch.tensor(sample_X, dtype=torch.float32)
        y = torch.tensor(sample_y, dtype=torch.float32)
        
        return X, y
