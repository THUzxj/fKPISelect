import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score


def fill_dataset(df):
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.fillna(0)
    return df


def get_default_device():
    import torch
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def plot_history(history):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()


def plot_history_save(history, fig_path):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.savefig(fig_path)


def histogram(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.hist([y_pred[y_test == 0],
              y_pred[y_test == 1]],
             bins=20,
             color=['#82E0AA', '#EC7063'], stacked=True)
    plt.title("Results", size=20)
    plt.grid()
    plt.show()


def ROC(y_test, y_pred, fig_path, fig_name="roc.png"):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.plot(fpr, 1-fpr, 'r:')
    plt.plot(fpr[idx], tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.savefig(os.path.join(fig_path, fig_name))
    return tr[idx]


def confusion_matrix(target, predicted, perc=False):

    data = {'y_Actual':    target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted', 'y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=[
                                   'Predicted'], colnames=['Actual'])

    if perc:
        sns.heatmap(confusion_matrix/np.sum(confusion_matrix),
                    annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()


def read_time_series(file_name):
    data = pd.read_csv(file_name)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp")
    data = data.sort_index()
    return data


def get_suffix_num(name):
    return int(name.split('-')[-1])


def get_suffix_num_v2(name):
    import re
    return int(re.split('_|-', name)[-1])


def read_results(path, label=None, anomaly_ratio=0.2, dataset_name=None):
    event_hit_data = []
    dirs = [dir for dir in os.listdir(
        path) if label in dir] if label is not None else os.listdir(path)
    dirs = sorted(dirs, key=get_suffix_num_v2)
    # print(dirs)
    for dir in dirs:
        if dataset_name is not None:
            dataset = dataset_name
        else:
            dataset = dir.split('_')[-1]
        file_path = os.path.join(
            path, dir, "events", f"event_results_{dataset}_{anomaly_ratio}.txt")
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue
        with open(file_path) as f:
            data = []
            for l in f:
                data.append(int(l))
        # print(file_path, len(data))
        event_hit_data.append(data)

    event_hit_data = np.array(event_hit_data).astype(np.int_)
    return event_hit_data


def read_f1(path, label=None, anomaly_ratio=0.2, dataset_name=None):
    f1s = []
    dirs = [dir for dir in os.listdir(
        path) if label in dir] if label is not None else os.listdir(path)
    dirs = sorted(dirs, key=get_suffix_num_v2)
    for dir in dirs:
        if dataset_name is not None:
            dataset = dataset_name
        else:
            dataset = dir.split('_')[-1]
        file_path = os.path.join(
            path, dir, "f1", f"f1_{dataset}_{anomaly_ratio}.csv")
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue

        f1 = pd.read_csv(file_path)
        f1s.append(f1)
    return f1s


def read_anomaly_scores(path, label=None, anomaly_ratio=0.2):
    anomaly_scores = []
    dirs = [dir for dir in os.listdir(
        path) if label in dir] if label is not None else os.listdir(path)
    dirs = sorted(dirs, key=get_suffix_num)
    print(dirs)
    for dir in dirs:
        dataset = dir.split('_')[-1]
        file_path = os.path.join(path, dir, "results",
                                 f"results_{dataset}_{anomaly_ratio}.csv")

        df = pd.read_csv(file_path)
        anomaly_scores.append(df['score'].values)
    anomaly_scores = np.array(anomaly_scores)
    print(anomaly_scores.shape)
    anomaly_scores = anomaly_scores.astype(np.float)
    return anomaly_scores
