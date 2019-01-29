import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import time
import os

from martins_stuff import storage_utils

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.num_epochs = None
        self.train_data = None
        self.optimizer = None
        self.train_file_path = None
        self.cross_entropy = None

        use_gpu = False

        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            self.device = torch.device('cuda')  # sets device to be cuda
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
            print("use GPU")
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU

    def get_acc_batch(self,x_batch,y_batch,y_batch_pred=None):
        """
        :param x_batch: array or tensor
        :param y_batch: array, one-hot-encoded
        :param y_batch_pred:  tensor, (because results from model forward pass)
        :return:
        """

        if type(x_batch) is np.ndarray:
            x_batch = torch.Tensor(x_batch).float().to(device=self.device)

        if y_batch_pred is None:
            y_batch_pred = self(x_batch)

        y_batch_int =  np.argmax(y_batch,axis=1)
        y_batch_int = torch.Tensor(y_batch_int).long().to(device=self.device)
        _, y_pred_batch_int = torch.max(y_batch_pred.data, 1)  # argmax of predictions
        acc = np.mean(list(y_pred_batch_int.eq(y_batch_int.data).cpu()))  # compute accuracy

        return acc

    def train_iter(self,x_train_batch,y_train_batch):
        """
        :param x_train_batch: array, one-hot-encoded
        :param y_train_batch: array, one-hot-encoded
        :return:
        """

        # CrossEntropyLoss. Input: (N,C), target: (N) each value is integer encoded.

        self.train()
        y_train_batch_int = np.argmax(y_train_batch,axis=1)
        y_train_batch_int = torch.Tensor(y_train_batch_int).long().to(device=self.device)
        x_train_batch = torch.Tensor(x_train_batch).float().to(device=self.device)
        y_pred_batch = self(x_train_batch) # model forward pass
        loss = F.cross_entropy(input=y_pred_batch,target=y_train_batch_int) # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc_batch = self.get_acc_batch(x_train_batch,y_train_batch,y_pred_batch)

        return loss.data, acc_batch

    def save_train_epoch_results(self, batch_statistics,train_file_path):
        statistics_to_save = {"train_acc":0, "train_loss":0, "epoch_train_time":0,"current_epoch":0}
        statistics_to_save["current_epoch"] = batch_statistics["current_epoch"]
        statistics_to_save["epoch_train_time"] = batch_statistics["epoch_train_time"]

        for key, value in batch_statistics.items():
            if key not in ["current_epoch","epoch_train_time"]:
                batch_values = np.array(batch_statistics[key])
                epoch_val = np.mean(batch_values)  # get mean of all metrics of current epoch metrics dict
                statistics_to_save[key] = np.around(epoch_val,decimals=4)

        print(statistics_to_save)
        storage_utils.save_statistics(statistics_to_save,train_file_path)

    def train_full(self, train_data, num_epochs, optimizer,train_file_path,model_save_dir):
        self.num_epochs = num_epochs
        self.train_data = train_data
        self.optimizer = optimizer
        self.train_file_path = train_file_path
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        for current_epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            batch_statistics = {"train_acc": [], "train_loss": [],"epoch_train_time":0}

            for i,(x_train_batch, y_train_batch) in enumerate(self.train_data):  # get data batches
                loss_batch, accuracy_batch = self.train_iter(x_train_batch, y_train_batch)  # take a training iter step
                batch_statistics["train_loss"].append(loss_batch)  # add current iter loss to the train loss list
                batch_statistics["train_acc"].append(accuracy_batch)  # add current iter acc to the train acc list

                print(i,end=' ')
            print('')

            batch_statistics["epoch_train_time"] = time.time() - epoch_start_time
            batch_statistics["current_epoch"] = current_epoch

            self.save_train_epoch_results(batch_statistics,train_file_path) # uses batch statistics to calculate full data stats.
            self.save_model(model_save_dir,current_epoch)

    def evaluate_full(self):
        # During training a model is saved. Here I want to load the model from every epoch and test it on a validation
        # set. Seperate trainng from evaluating. I like this better.

        pass

    def save_model(self, model_save_dir,current_epoch):
        state = dict()
        state['network'] = self.state_dict()  # save network parameter and other variables.
        model_save_name = "model"
        model_path = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(current_epoch)))

        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(state, f=model_path)

    def load_model(self, model_path):
        state = torch.load(f=model_path)
        self.load_state_dict(state_dict=state['network'])

def main():
    # test experiment builder
    pass

if __name__ == '__main__':
    main()