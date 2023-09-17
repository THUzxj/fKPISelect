import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BaseDataset:
    def __init__(self, data_path, scalar_class=StandardScaler) -> None:
        self._load_data(data_path, scalar_class)

    def _load_data(self, data_path, scalar_class):
        self.scaler = scalar_class()
        data = np.load(data_path + "/train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/labels.npy")
        self.train_labels = np.zeros(self.train.shape[0])
        print(self.train.shape)

    def _select_kpis(self, select_file):
        selected_kpis = np.loadtxt(select_file, dtype=np.int32)
        print(selected_kpis)
        self.train = self.train[:, selected_kpis]
        self.test = self.test[:, selected_kpis]

    def describe(self):
        print("train shape:", self.train.shape)
        print("test shape:", self.test.shape)
        print("test labels shape:", self.test_labels.shape)


class CSVDataset(BaseDataset):
    def _load_data(self, data_path, scalar_class):
        self.scaler = scalar_class()
        data = pd.read_csv(data_path + "/train.csv", index_col=0).values
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + "/test.csv", index_col=0).values
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(
            data_path + "/labels.csv", index_col=0).values.reshape(-1)
        self.train_labels = np.zeros(self.train.shape[0])
        print(self.train.shape)


class SelectDataset(BaseDataset):
    def __init__(self, data_path, select_file, scalar_class=StandardScaler):
        self._load_data(data_path, scalar_class)
        self._select_kpis(select_file)


class SelectCSVDataset(CSVDataset):
    def __init__(self, data_path, select_file, scalar_class=StandardScaler):
        self._load_data(data_path, scalar_class)
        self._select_kpis(select_file)


def get_dataset_v2(data_path, dataset, scalar_class, select_file=None):
    if (dataset == 'SMD'):
        dataset = BaseDataset(data_path, scalar_class)
    elif (dataset == 'WADI'):
        dataset = BaseDataset(data_path, scalar_class)
    elif (dataset == 'MUL'):
        dataset = BaseDataset(data_path, scalar_class)
    elif (dataset.startswith("SMDSELECT")):
        dataset = SelectDataset(data_path, select_file, scalar_class)
    elif (dataset == 'PROMEV2'):
        dataset = CSVDataset(data_path, scalar_class)
    elif (dataset.startswith('PROMESELECTV2')):
        dataset = SelectCSVDataset(data_path, select_file, scalar_class)
    elif (dataset == 'special'):
        dataset = BaseDataset(data_path, scalar_class)
    elif (dataset == 'synthetic'):
        dataset = BaseDataset(data_path, scalar_class)
    else:
        dataset = BaseDataset(data_path, scalar_class)
    return dataset
