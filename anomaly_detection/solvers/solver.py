import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from anomaly_detection.data_factory.dataset import get_dataset
from anomaly_detection.data_factory.data_loader import SegLoader
from anomaly_detection.solvers.utils import write_event_results, write_results, write_info
from anomaly_detection.metric import calc_anomaly_event, point_adjustment, calc_event_f1


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, checkpoint_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.checkpoint_name = checkpoint_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(
            path, str(self.checkpoint_name) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class EarlyStoppingBsad:
    def __init__(self, patience=7, verbose=False, checkpoint_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_name = checkpoint_name

    def __call__(self, val_loss,  model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(
            path, str(self.checkpoint_name) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


def statistics_single_threshold_point_adjustment(test_energy, thresh, test_labels):
    pred = (test_energy > thresh).astype(int)
    origin_pred = pred.copy()

    gt = test_labels.astype(int)
    # detection adjustment
    # labels
    pred = point_adjustment(gt, pred)

    anomaly_event_hit = calc_anomaly_event(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(
        gt, pred, average='binary')

    f1_data = pd.DataFrame(
        {
            'precision': [precision],
            'recall': [recall],
            'f1': [f_score]
        }
    )

    return origin_pred, accuracy, f1_data, anomaly_event_hit


def statistics_metrics(pred, test_labels):
    origin_pred = pred.copy()
    gt = test_labels.astype(int)

    # Calculate F1 with origin prediction and labels
    origin_precision, origin_recall, origin_f_score, support = precision_recall_fscore_support(gt, origin_pred,
                                                                                               average='binary')

    # detection adjustment
    # labels
    pred = point_adjustment(gt, pred)

    anomaly_event_hit = calc_anomaly_event(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)

    # Calculate F1 with point adjustment method
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(
        gt, pred, average='binary')

    # Calculate F1 with event view
    event_precision, event_recall, event_f1 = calc_event_f1(gt, origin_pred)

    f1_data = pd.DataFrame(
        {
            'precision': [origin_precision, precision, event_precision],
            'recall': [origin_recall, recall, event_recall],
            'f1': [origin_f_score, f_score, event_f1]
        }
    )

    return accuracy, f1_data, anomaly_event_hit


class Solver(object):
    def __init__(self, config) -> None:
        # default values
        self.select_file = None
        self.inspect_scores = False
        self.multiple_anomaly_ratios = False
        self.model_init_checkpoint = None
        self.save_output = False

        self.__dict__.update(config)
        print('config: ', config)

        if self.scaler == "standard":
            self.scaler_class = StandardScaler
        elif self.scaler == "minmax":
            self.scaler_class = MinMaxScaler

        if self.data_path:
            dataset_object = get_dataset(
                self.data_path, self.dataset, self.scaler_class)
            self.load_dataset_object(dataset_object)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.criterion = nn.MSELoss()
        self.model_loaded = False

    def load_dataset_object(self, dataset_object):
        self.dataset_object = dataset_object
        self.train_loader = DataLoader(dataset=SegLoader(self.dataset_object, win_size=self.win_size, step=1, mode="train"),
                                       batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.vali_loader = DataLoader(dataset=SegLoader(self.dataset_object, win_size=self.win_size, step=1, mode="val"),
                                      batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(dataset=SegLoader(self.dataset_object, win_size=self.win_size, step=1, mode="test"),
                                      batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.thre_loader = DataLoader(dataset=SegLoader(self.dataset_object, win_size=self.win_size, step=1, mode="thre"),
                                      batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.train_input_loader = DataLoader(dataset=SegLoader(self.dataset_object, win_size=self.win_size, step=1, mode="thre_on_train"),
                                             batch_size=self.batch_size, shuffle=False, num_workers=0)

    def build_model(self):
        pass

    def vali(self):
        pass

    def train(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        early_stopping = EarlyStoppingBsad(
            patience=self.patience, verbose=True, checkpoint_name=self.model_name)

        for epoch in range(self.num_epochs):

            self.model.train()

            for i, (batch, labels) in enumerate(tqdm(self.train_loader)):

                loss = self.train_epoch(batch, labels)

                if (i + 1) % self.log_step == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {}".format(epoch +
                          1, self.num_epochs, i + 1, len(self.train_loader), loss))

            vali_loss = self.vali()
            self.report_loss(epoch, loss, vali_loss)
            early_stopping(vali_loss, self.model, self.model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def train_epoch(self, batch, labels):
        pass

    def test(self):
        pass

    def test_on_train_data(self):
        pass

    def report_loss(self, epoch, train_loss, vali_loss):
        print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} ".format(
                epoch + 1, train_loss, vali_loss))

        with SummaryWriter(os.path.join(self.output_dir, "tensorboard")) as w:
            w.add_scalar('Train Loss', train_loss, epoch)
            w.add_scalar('Vali Loss', vali_loss, epoch)

    def load_model(self):
        if self.model_loaded == False:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.model_name) + '_checkpoint.pth')))
            self.model.eval()
            self.model_loaded = True

    def _statistics(self, test_scores, thre_scores, test_labels):
        return self._statistics_single_percentage(
            test_scores, thre_scores, test_labels, self.anomaly_ratio)

    def _statistics_single_percentage(self, test_scores, thre_scores, test_labels, anomaly_ratio):
        thresh = np.percentile(thre_scores, 100 - anomaly_ratio)
        pred = (test_scores > thresh).astype(int)
        accuracy, f1_data, anomaly_event_hit = statistics_metrics(
            pred, test_labels)
        print("anomaly_ratio: ", anomaly_ratio,
              "threshold: ", thresh, "accuracy: ", accuracy)
        print(f1_data)

        if self.save_output:
            os.makedirs(os.path.join(
                self.output_dir, "results"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "events"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "f1"), exist_ok=True)
            output_file = os.path.join(
                self.output_dir, "results", "results_{}_{}.csv".format(self.dataset, anomaly_ratio))
            event_output_file = os.path.join(
                self.output_dir, "events", "event_results_{}_{}.txt".format(self.dataset, anomaly_ratio))

            write_results(test_labels, pred, test_scores, output_file)
            write_event_results(anomaly_event_hit, event_output_file)

            f1_data.to_csv(os.path.join(
                self.output_dir, "f1", "f1_{}_{}.csv".format(self.dataset, anomaly_ratio)))

        return thresh, pred
