import torch
import torch.nn as nn
import numpy as np
import os
import time


from anomaly_detection.models.AnomalyTransformer import AnomalyTransformer

from anomaly_detection.solvers.solver import Solver, EarlyStopping, adjust_learning_rate


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class AnomalyTransformerSolver(Solver):
    def __init__(self, config):
        super(AnomalyTransformerSolver, self).__init__(config)

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def get_model(self):
        return self.model

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                           self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, checkpoint_name=self.model_name)
        train_steps = len(self.train_loader)

        if (self.model_init_checkpoint):
            checkpoint = torch.load(self.model_init_checkpoint)
            self.model.load_state_dict(checkpoint)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            self.model.train()

            epoch_time = time.time()

            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * \
                        ((self.num_epochs - epoch) * train_steps - i)
                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # self._test_performance(self.thre_loader)
            torch.cuda.empty_cache()
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def _calculate_metric(self, series, prior, temperature):
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                           self.win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                           self.win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size)),
                    series[u].detach()) * temperature
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        return metric.detach().cpu().numpy()

    def _calculate_energy(self, criterion, temperature, data_loader):
        attens_energy = []
        if self.inspect_scores:
            reconstruction_loss_per_kpi = []
            attens_energy_per_kpi = []
            inputs = []
            outputs = []

        for i, (input_data, labels) in enumerate(data_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

            if self.inspect_scores:
                loss_per_kpi = criterion(input, output)
                reconstruction_loss_per_kpi.append(
                    loss_per_kpi.detach().cpu().numpy())
                # np.matmul(metric, loss_per_kpi, axes=([1], [1]))
                cri_per_kpi = metric.unsqueeze(
                    1) * loss_per_kpi.transpose(1, 2)
                attens_energy_per_kpi.append(
                    cri_per_kpi.detach().cpu().numpy())
                inputs.append(input.detach().cpu().numpy())
                outputs.append(output.detach().cpu().numpy())
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        if self.inspect_scores and self.mode == "test":
            reconstruction_loss_per_kpi = np.concatenate(
                reconstruction_loss_per_kpi, axis=0)
            reconstruction_loss_per_kpi = reconstruction_loss_per_kpi.reshape(
                -1, reconstruction_loss_per_kpi.shape[-1])
            attens_energy_per_kpi = np.concatenate(
                attens_energy_per_kpi, axis=0).transpose((0, 2, 1))
            attens_energy_per_kpi = attens_energy_per_kpi.reshape(
                -1, attens_energy_per_kpi.shape[-1])
            np.save(os.path.join(
                self.output_dir, "reconstruction_loss_per_kpi.npy"), reconstruction_loss_per_kpi)
            np.save(os.path.join(self.output_dir,
                    "attens_energy_per_kpi.npy"), attens_energy_per_kpi)
            np.save(os.path.join(self.output_dir,
                    "attens_energy.npy"), attens_energy)

            inputs = np.concatenate(inputs, axis=0)
            outputs = np.concatenate(outputs, axis=0)
            print("input output shape:", inputs.shape, outputs.shape)
            np.save(os.path.join(self.output_dir, "inputs.npy"), inputs)
            np.save(os.path.join(self.output_dir, "outputs.npy"), outputs)
        return np.array(attens_energy)

    def _test_performance(self, loader):
        with torch.no_grad():
            self.model.eval()
            temperature = 50
            criterion = nn.MSELoss(reduction='none')

            # (1) stastic on the train set
            train_energy = self._calculate_energy(
                criterion, temperature, self.train_loader)

            # (2) find the threshold
            test_energy = self._calculate_energy(
                criterion, temperature, self.thre_loader)

            combined_energy = np.concatenate(
                [train_energy, test_energy], axis=0)

            # (3) evaluation on the test set
            test_labels = []
            for i, (input_data, labels) in enumerate(loader):
                test_labels.append(labels)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            test_energy = self._calculate_energy(
                criterion, temperature, loader)
            thresh, pred = self._statistics(
                test_energy, combined_energy, test_labels)
        return test_labels, test_energy, thresh, pred

    def _test(self, loader):
        self.load_model()
        print("======================TEST MODE======================")
        start_time = time.time()
        self._test_performance(loader)
        end_time = time.time()
        print("test time:", round(end_time - start_time, 4))

    def test(self):
        self._test(self.thre_loader)

    def test_on_train_data(self):
        self._test(self.thre_on_train_loader)

    def _calculate_energy_only_reconstruction(self, criterion, temperature, data_loader):
        attens_energy = []
        for i, (input_data, labels) in enumerate(data_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            cri = torch.mean(loss, dim=1)
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        return np.array(attens_energy)

    def test_only_reconstruction(self):
        self.load_model()
        print("======================TEST MODE======================")
        with torch.no_grad():
            self.model.eval()
            temperature = 50
            criterion = nn.MSELoss(reduction='none')

            # (1) stastic on the train set
            train_energy = self._calculate_energy_only_reconstruction(
                criterion, temperature, self.train_loader)

            # (2) find the threshold
            test_energy = self._calculate_energy_only_reconstruction(
                criterion, temperature, self.test_loader)

            combined_energy = np.concatenate(
                [train_energy, test_energy], axis=0)

            # (3) evaluation on the test set
            test_labels = []
            for i, (input_data, labels) in enumerate(self.test_loader):
                test_labels.append(labels)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            # test_energy = self._calculate_energy_only_reconstruction(criterion, temperature, self.test_loader)
            self._statistics(test_energy, combined_energy, test_labels)

    def save_train_input_output(self):
        with torch.no_grad():
            self.model.eval()
            inputs = []
            outputs = []
            metrics = []
            temperature = 50

            for i, (input_data, labels) in enumerate(self.train_input_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                metric = self._calculate_metric(series, prior, temperature)

                inputs.append(input.detach().cpu().numpy())
                outputs.append(output.detach().cpu().numpy())
                metrics.append(metric)

        inputs = np.concatenate(inputs, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        print("input output shape:", inputs.shape, outputs.shape)
        np.save(os.path.join(self.output_dir, "train_inputs.npy"), inputs)
        np.save(os.path.join(self.output_dir, "train_outputs.npy"), outputs)

        metrics = np.concatenate(metrics, axis=0)
        np.save(os.path.join(self.output_dir, "train_metrics.npy"), metrics)

    def save_test_input_output(self):

        with torch.no_grad():
            self.model.eval()
            inputs = []
            outputs = []
            all_labels = []
            metrics = []

            temperature = 50

            for i, (input_data, labels) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                metric = self._calculate_metric(series, prior, temperature)

                inputs.append(input.detach().cpu().numpy())
                outputs.append(output.detach().cpu().numpy())
                all_labels.append(labels)
                metrics.append(metric)

        inputs = np.concatenate(inputs, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        print("input output shape:", inputs.shape, outputs.shape)
        np.save(os.path.join(self.output_dir, "test_inputs.npy"), inputs)
        np.save(os.path.join(self.output_dir, "test_outputs.npy"), outputs)
        all_labels = np.concatenate(all_labels, axis=0)
        print(all_labels.shape)
        all_labels = np.any(np.array(all_labels) == 1, axis=1)
        print(all_labels.shape)
        np.save(os.path.join(self.output_dir, "test_labels.npy"), all_labels)

        metrics = np.concatenate(metrics, axis=0)
        print(metrics.shape)
        np.save(os.path.join(self.output_dir, "test_metrics.npy"), metrics)

    def load_model(self):
        if self.model_loaded == False:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.model_name) + '_checkpoint.pth')))
            self.model.eval()
            self.model_loaded = True

    def feature_latent(self):
        # self.load_model()
        latents = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output = self.model.encode(input)
            latents.append(output.detach().cpu().numpy())
        latents = np.concatenate(latents, axis=0)
        latents = latents.mean(axis=1)
        feature_latent = latents.mean(axis=0)
        return feature_latent
