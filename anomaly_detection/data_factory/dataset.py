import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Dataset:
    def __init__(self):
        self._load_data()

    def _load_data(self):
        pass


class ArrayDataset(Dataset):
    def __init__(self, data_path, scaler_class=StandardScaler) -> None:
        self._load_data(data_path, scaler_class)

    def _load_data(self, data_path, scaler_class):
        self.scaler = scaler_class()
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

    def select_kpis(self, selected_kpis):
        self.train = self.train[:, selected_kpis]
        self.test = self.test[:, selected_kpis]

    def describe(self):
        print("train shape:", self.train.shape)
        print("test shape:", self.test.shape)
        print("test labels shape:", self.test_labels.shape)


class CSVDataset(ArrayDataset):
    def _load_data(self, data_path, scaler_class):
        self.scaler = scaler_class()
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


def get_dataset(data_path, dataset, scaler_class):
    if (dataset == 'SMD'):
        dataset = ArrayDataset(data_path, scaler_class)
    elif (dataset == 'PROMEV2'):
        dataset = CSVDataset(data_path, scaler_class)
    elif (dataset == 'synthetic'):
        dataset = ArrayDataset(data_path, scaler_class)
    else:
        dataset = ArrayDataset(data_path, scaler_class)
    return dataset
