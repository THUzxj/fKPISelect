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

from anomaly_detection.data_factory.dataset import get_dataset_v2
from anomaly_detection.data_factory.data_loader import SegLoader
from anomaly_detection.solvers.utils import write_event_results, write_results, write_info
from anomaly_detection.metric import calc_anomaly_event, point_adjustment, my_event_f1


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


def statistics_single_threshold(test_energy, thresh, test_labels):
    pred = (test_energy > thresh).astype(int)
    origin_pred = pred.copy()

    gt = test_labels.astype(int)

    # print("pred:   ", pred.shape)
    # print("gt:     ", gt.shape)

    # detection adjustment
    # labels
    pred = point_adjustment(gt, pred)

    anomaly_event_hit = calc_anomaly_event(gt, pred)
    # if event_output_file:
    #     write_event_results(anomaly_event_hit, event_output_file)

    pred = np.array(pred)
    gt = np.array(gt)
    # print("pred: ", pred.shape)
    # print("gt:   ", gt.shape)

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(
        gt, pred, average='binary')
    # print(
    #     "adjusted: Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F1-score : {:0.4f} ".format(
    #         accuracy, precision,
    #         recall, f_score))

    origin_precision, origin_recall, origin_f_score, support = precision_recall_fscore_support(gt, origin_pred,
                                                                                               average='binary')
    # print(
    #     "Origin: Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F1-score : {:0.4f} ".format(
    #         accuracy, origin_precision,
    #         origin_recall, origin_f_score))

    if origin_precision + recall == 0:
        my_f1_score = 0
    else:
        my_f1_score = 2 * (origin_precision * recall) / \
            (origin_precision + recall)

    # print("My Point Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F1-score : {:0.4f}".format(
    #     accuracy, origin_precision, recall, my_f1_score))

    event_precision, event_recall, event_f1 = my_event_f1(gt, origin_pred)

    # if(output_file):
    #     write_results(gt, pred, test_energy, output_file)

    f1_data = pd.DataFrame(
        {
            'precision': [origin_precision, precision, event_precision],
            'recall': [recall, recall, event_recall],
            'f1': [my_f1_score, f_score, event_f1]
        }
    )

    return origin_pred, accuracy, f1_data, anomaly_event_hit


class Solver(object):
    def __init__(self, config) -> None:
        # default values
        self.select_file = None
        self.inspect_scores = False
        self.multiple_anomaly_ratios = False
        self.model_init_checkpoint = None

        self.__dict__.update(config)
        print('config: ', config)

        if self.scaler == "standard":
            scaler_class = StandardScaler
        elif self.scaler == "minmax":
            scaler_class = MinMaxScaler

        self.dataset_loader = get_dataset_v2(
            self.data_path, self.dataset, scaler_class, self.select_file)
        self.train_loader = DataLoader(dataset=SegLoader(self.dataset_loader, win_size=self.win_size, step=1, mode="train"),
                                       batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.vali_loader = DataLoader(dataset=SegLoader(self.dataset_loader, win_size=self.win_size, step=1, mode="val"),
                                      batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(dataset=SegLoader(self.dataset_loader, win_size=self.win_size, step=1, mode="test"),
                                      batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.thre_loader = DataLoader(dataset=SegLoader(self.dataset_loader, win_size=self.win_size, step=1, mode="thre"),
                                      batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.train_input_loader = DataLoader(dataset=SegLoader(self.dataset_loader, win_size=self.win_size, step=1, mode="thre_on_train"),
                                             batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.criterion = nn.MSELoss()
        self.model_loaded = False

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
        if self.multiple_anomaly_ratios:
            # accuracy, precision, recall, f_score = self._statistics_multi_threshold_binary_search(
            #     test_scores, thre_scores, test_labels)
            accuracy, precision, recall, f_score = self._statistics_multi_threshold(
                test_scores, thre_scores, test_labels)
        else:
            self._statistics_single_percentage(
                test_scores, thre_scores, test_labels, self.anomaly_ratio)

    def _statistics_multi_threshold(self, test_scores, thre_scores, test_labels):
        os.makedirs(self.output_dir, exist_ok=True)

        # anomaly_ratios = np.array([0.025, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0])
        # anomaly_ratios = np.arange(0, 3, step=0.05)
        anomaly_ratios = np.array(
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        # anomaly_ratios = np.concatenate([anomaly_ratios, np.array([5, 10, 20, 30])])
        print("anomaly_ratios: ", anomaly_ratios)
        results = []
        f1_datas = []
        print("thre_scores: ", thre_scores)
        os.makedirs(os.path.join(self.output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "events"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "f1"), exist_ok=True)

        for anomaly_ratio in anomaly_ratios:
            print("anomaly_ratio: ", anomaly_ratio)
            thresh = np.percentile(thre_scores, 100 - anomaly_ratio)
            print("Threshold :", thresh)

            output_file = os.path.join(
                self.output_dir, "results", "results_{}_{}.txt".format(self.dataset, anomaly_ratio))
            event_output_file = os.path.join(
                self.output_dir, "events", "event_results_{}_{}.txt".format(self.dataset, anomaly_ratio))

            pred, accuracy, f1_data, anomaly_event_hit = statistics_single_threshold(
                test_scores, thresh, test_labels)
            print(test_labels.shape, pred.shape, test_scores.shape)
            write_results(test_labels.reshape(-1), pred,
                          test_scores, output_file)
            write_event_results(anomaly_event_hit, event_output_file)

            f1_data.to_csv(os.path.join(
                self.output_dir, "f1", "f1_{}_{}.txt".format(self.dataset, anomaly_ratio)))

            # print("anomaly_ratio: ", anomaly_ratio, "threshold: ", thresh, "accuracy: ",
            #       accuracy, "precision: ", precision, "recall: ", recall, "f_score: ", f_score)

            results.append(
                [float(anomaly_ratio), float(thresh), float(accuracy)])
            f1_datas.append(f1_data)

        # print(results)
        results = np.array(results, dtype=object)

        f1_datas_values = np.array([f1_data.values for f1_data in f1_datas])

        adjust_index = np.argmax(f1_datas_values[:, 1, 2])
        f1_datas[adjust_index].to_csv(os.path.join(
            self.output_dir, "adjusted_max_f1.txt"))
        anomaly_ratio, thresh, accuracy = results[adjust_index]
        write_info(accuracy, anomaly_ratio, thresh, f1_datas_values[adjust_index][1][0], f1_datas_values[adjust_index]
                   [1][1], f1_datas_values[adjust_index][1][2], os.path.join(self.output_dir, "adjusted_info.txt"))

        my_index = np.argmax(f1_datas_values[:, 0, 2])
        f1_datas[my_index].to_csv(os.path.join(
            self.output_dir, "my_max_f1.txt"))
        anomaly_ratio, thresh, accuracy = results[my_index]
        write_info(accuracy, anomaly_ratio, thresh, f1_datas_values[my_index][0][0], f1_datas_values[my_index]
                   [0][1], f1_datas_values[my_index][0][2], os.path.join(self.output_dir, "my_info.txt"))

        event_index = np.argmax(f1_datas_values[:, 2, 2])
        print(f1_datas_values[:, 2, 2])
        f1_datas[event_index].to_csv(os.path.join(
            self.output_dir, "event_max_f1.txt"))
        anomaly_ratio, thresh, accuracy = results[event_index]
        write_info(accuracy, anomaly_ratio, thresh, f1_datas_values[event_index][2][0], f1_datas_values[event_index]
                   [2][1], f1_datas_values[event_index][2][2], os.path.join(self.output_dir, "event_info.txt"))

        return accuracy, f1_datas[event_index]["precision"][2], f1_datas[event_index]["recall"][2], f1_datas[event_index]["f1"][2]

    def _statistics_single_percentage(self, test_scores, thre_scores, test_labels, anomaly_ratio):
        thresh = np.percentile(thre_scores, 100 - anomaly_ratio)
        os.makedirs(os.path.join(self.output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "events"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "f1"), exist_ok=True)
        output_file = os.path.join(
            self.output_dir, "results", "results_{}_{}.txt".format(self.dataset, anomaly_ratio))
        event_output_file = os.path.join(
            self.output_dir, "events", "event_results_{}_{}.txt".format(self.dataset, anomaly_ratio))

        pred, accuracy, f1_data, anomaly_event_hit = statistics_single_threshold(
            test_scores, thresh, test_labels)
        print("anomaly_ratio: ", anomaly_ratio,
              "threshold: ", thresh, "accuracy: ", accuracy)
        print(f1_data)

        # write_results(test_labels, pred, test_scores, output_file)
        write_event_results(anomaly_event_hit, event_output_file)

        f1_data.to_csv(os.path.join(
            self.output_dir, "f1", "f1_{}_{}.txt".format(self.dataset, anomaly_ratio)))

    def _statistics_multi_threshold_binary_search(self, test_scores, thre_scores, test_labels):
        os.makedirs(self.output_dir, exist_ok=True)

        lower_bound = 0.0
        upper_bound = 100.0
        max_iterations = 50
        tolerance = 0.01

        result = None

        # Perform a binary search to find the threshold that maximizes the F1 score
        for i in range(max_iterations):
            # Calculate the midpoint
            midpoint = (lower_bound + upper_bound) / 2

            accuracy, precision, recall, f_score, adjusted_data = self._statistics_single_percentage(
                test_scores, thre_scores, test_labels, midpoint)

            result = [midpoint, accuracy, precision,
                      recall, f_score, adjusted_data]

            accuracy_2, precision_2, recall_2, f_score_2, adjusted_data = self._statistics_single_percentage(
                test_scores, thre_scores, test_labels, midpoint + tolerance)

            i = 2
            while f_score == f_score_2:
                accuracy_2, precision_2, recall_2, f_score_2, adjusted_data = self._statistics_single_percentage(
                    test_scores, thre_scores, test_labels, midpoint + tolerance * i)
                i += 1

            if f_score > f_score_2:
                upper_bound = midpoint
            else:
                lower_bound = midpoint

            if abs(upper_bound - lower_bound) < tolerance:
                break
        # print("max f1: ")
        # threshold = np.percentile(thre_scores, 100 - result[0])
        # print("anomaly_ratio: ", result[0], "threshold: ", threshold, "accuracy: ",
        #       accuracy, "precision: ", precision, "recall: ", recall, "f_score: ", f_score)
        # write_scores(accuracy, precision, recall, f_score, result[0], os.path.join(
        #     self.output_dir, "scores_binary_{}.txt".format(self.dataset)))

        # with open(os.path.join(
        #         self.output_dir, "scores_binary_{}.txt".format(self.dataset)), 'a') as f:
        #     f.write(
        #         f"{accuracy},{adjusted_data[0]},{adjusted_data[1]},{adjusted_data[2]}\n")
        return accuracy, precision, recall, f_score
