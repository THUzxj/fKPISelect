import csv
import pandas as pd
import numpy as np
import datetime


def load_fi_log(filepath):
    f = open(filepath)
    fis = []
    reader = csv.reader(f)
    for row in reader:
        fis.append(row)
    return fis


def load_data(filepath):
    return pd.read_csv(filepath)


def generate_label(fi_logs, data):
    data["label"] = np.zeros(len(data))

    for i in range(int(len(fi_logs)/2)):
        start_time = datetime.strptime(fi_logs[2*i][0], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(fi_logs[2*i+1][0], "%Y-%m-%d %H:%M:%S")

        data["label"][start_time:end_time] = 1
    return data
