import torch
import os
import time

from anomaly_detection.models.usad import *
from anomaly_detection.utils import to_device
from anomaly_detection.solvers.solver import Solver, EarlyStopping, adjust_learning_rate
from anomaly_detection.solvers.utils import write_event_results, write_results

device = get_default_device()

def evaluate(model, val_loader, n, w_size):
    outputs = []
    for (batch, labels) in val_loader:
        batch=to_device(batch,device)
        batch = batch.view([batch.shape[0], w_size])
        outputs.append(model.validation_step(batch, n))
    return model.validation_epoch_end(outputs)
    
def testing(model, test_loader, w_size, alpha=.5, beta=.5):
    results=[]
    for [batch] in test_loader:
        batch=to_device(batch,device)
        batch = batch.view([batch.shape[0], w_size])
        w1=model.decoder1(model.encoder(batch))
        w2=model.decoder2(model.encoder(w1))
        results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results


class UsadSolver(Solver):
    def __init__(self, config):
        super(UsadSolver, self).__init__(config)
        # self.build_model()

    def build_model(self):
        self.w_size=self.win_size*self.input_c
        self.z_size=self.win_size*self.hidden_size
        print("w_size: ", self.w_size, "z_size: ", self.z_size)
        self.model = UsadModel(self.w_size, self.z_size)
        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        pass
    
    def train(self):
        opt_func = torch.optim.Adam
        print(self.model)
        print("======================TRAIN MODE======================")
        history = []
        optimizer1 = opt_func(list(self.model.encoder.parameters())+list(self.model.decoder1.parameters()))
        optimizer2 = opt_func(list(self.model.encoder.parameters())+list(self.model.decoder2.parameters()))

        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, checkpoint_name=self.model_name)

        os.makedirs(self.model_save_path, exist_ok=True)
        for epoch in range(self.num_epochs):

            epoch_time = time.time()
            self.model.train()
            for (batch, label) in self.train_loader:
                batch=to_device(batch,device)
                batch = batch.view([batch.shape[0], self.w_size])
                #Train AE1
                loss1,loss2 = self.model.training_step(batch,epoch+1)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                
                
                #Train AE2
                loss1,loss2 = self.model.training_step(batch,epoch+1)
                loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                
            result = evaluate(self.model, self.test_loader, epoch+1, self.w_size)

            early_stopping(result["val_loss1"], result["val_loss2"], self.model, path)
            self.model.epoch_end(epoch, result)
            self._test_performance(self.test_loader)
            history.append(result)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return history

        # history = training(200, self.model, self.train_loader, self.vali_loader, self.w_size, checkpoint_path = self.model_save_path)
        # plot_history_save(history, os.path.join(self.output_file, "train.png"))

    def _calculate_score(self, data_loader, alpha, beta):
        results = []
        if self.inspect_scores:
            inputs = []
            outputs_w1 = []
            outputs_w2 = []
            results_per_kpi = []

        for i, (input_data, labels) in enumerate(data_loader):
            input_data = to_device(input_data, device)
            input_data = input_data.view([input_data.shape[0], self.w_size])
            w1 = self.model.decoder1(self.model.encoder(input_data))
            w2 = self.model.decoder2(self.model.encoder(w1))
            results.append(alpha*torch.mean((input_data-w1)**2,axis=1)+beta*torch.mean((input_data-w2)**2,axis=1))
            if self.inspect_scores:
                output = alpha*(input_data-w1)**2+beta*(input_data-w2)**2
                results_per_kpi.append(output)
                inputs.append(input_data)
                outputs_w1.append(w1)
                outputs_w2.append(w2)
        results = torch.cat(results, dim=0)
        if self.inspect_scores and self.mode == "test":
            results_per_kpi = torch.cat(results_per_kpi, dim=0)
            results_per_kpi = results_per_kpi.view(-1, self.win_size, self.input_c)
            results_per_kpi = torch.mean(results_per_kpi, dim=1)
            np.save(os.path.join(self.output_dir, "results_per_kpi.npy"), results_per_kpi.cpu().detach().numpy())
            
            inputs = torch.cat(inputs, dim=0)
            outputs_w1 = torch.cat(outputs_w1, dim=0)
            outputs_w2 = torch.cat(outputs_w2, dim=0)
            np.save(os.path.join(self.output_dir, "inputs.npy"), inputs.cpu().detach().numpy())
            np.save(os.path.join(self.output_dir, "outputs_w1.npy"), outputs_w1.cpu().detach().numpy())
            np.save(os.path.join(self.output_dir, "outputs_w2.npy"), outputs_w2.cpu().detach().numpy())

        return results.cpu().detach().numpy()
    
    def _test_performance(self, loader):
        self.model.eval()

        alpha = .5
        beta = .5
        # (1) statisitc on the train set
        train_score = self._calculate_score(self.train_loader, alpha, beta)

        # (2) find the threshold
        test_score = self._calculate_score(self.test_loader, alpha, beta)

        combined_energy = np.concatenate([train_score, test_score], axis=0)


        # (3) evaluation on test data
        test_labels = []
        results = self._calculate_score(loader, alpha, beta)
        for i, (input_data, labels) in enumerate(loader):
            test_labels.append(labels)
        test_labels = torch.cat(test_labels, dim=0).detach().cpu().numpy()
        # test_labels = np.any(test_labels == 1, axis=1)
        test_labels = test_labels[:, 0]
        # test_labels = loader.dataset.train_labels[:results.shape[0]]

        self._statistics(results, combined_energy, test_labels)

    def _test(self, loader):
        print(self.model)
        checkpoint_path = os.path.join(self.model_save_path, str(self.model_name) + '_checkpoint.pth')
        self.model.load_state_dict(
            torch.load(
            checkpoint_path
            )
        )
        self.model.eval()
        start_time = time.time()
        self._test_performance(loader)
        end_time = time.time()
        print("test time:", round(end_time - start_time, 4))

    def test(self):
        return self._test(self.test_loader)

    def test_on_train_data(self):
        return self._test(self.train_loader)
    
    