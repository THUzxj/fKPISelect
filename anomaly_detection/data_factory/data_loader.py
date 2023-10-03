import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SegLoader:
    def __init__(self, dataset, win_size, step, mode="train"):
        self.dataset = dataset
        self.mode = mode
        self.step = step
        self.win_size = win_size

    def __len__(self):
        if self.mode == "train":
            return (self.dataset.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.dataset.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.dataset.test.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == "thre"):
            return (self.dataset.test.shape[0] - self.win_size) // self.win_size + 1
        elif (self.mode == "thre_on_train"):
            return (self.dataset.train.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.dataset.train[index:index + self.win_size]), np.float32(self.dataset.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.dataset.val[index:index + self.win_size]), np.float32(self.dataset.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.dataset.test[index:index + self.win_size]), np.float32(
                self.dataset.test_labels[index:index + self.win_size])
        elif (self.mode == "thre"):
            return np.float32(self.dataset.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.dataset.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        elif (self.mode == "thre_on_train"):
            return np.float32(self.dataset.train[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.dataset.train_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class TimeSeriesLoader:
    """
    Only test dataset
    """

    def __init__(self, time_series, labels, win_size, step):
        self.time_series = time_series
        self.labels = labels
        self.step = step
        self.win_size = win_size

    def __len__(self):
        return (self.time_series.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.time_series[
            index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
            self.labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
