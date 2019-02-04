from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider, ModifyDataProvider
from martins_stuff.ModelBuilder.simple_fnn import *
import numpy as np
import globals
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv

os.environ['MLP_DATA_DIR'] = 'data'


class Experiment(object):
    @staticmethod
    def _train(model, model_title, train_data, num_epochs, optimizer):
        saved_models_dir = os.path.join(globals.ROOT_DIR, 'SavedModels/' + model_title)
        train_results_file = os.path.join(globals.ROOT_DIR, 'ExperimentResults/' + model_title + '.txt')
        model.train_full(train_data, num_epochs, optimizer, train_results_file, saved_models_dir)
        with open('ExperimentResults/' + model_title + '.txt', 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            train_acc, train_loss = last_line.split('\t')[1:3]
        return float(train_acc), float(train_loss)

    @staticmethod
    def _evaluate(model, model_title, valid_data, num_epochs):
        saved_models_dir = os.path.join(globals.ROOT_DIR, 'SavedModels/' + model_title)
        eval_results_file = os.path.join(globals.ROOT_DIR, 'ExperimentResults/' + model_title + '_eval.txt')
        model.evaluate_full(valid_data, num_epochs, saved_models_dir, eval_results_file)

        with open('ExperimentResults/' + model_title + '_eval.txt', 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            valid_acc, valid_loss = last_line.split('\t')[-2:]
        return float(valid_acc), float(valid_loss)

    def _compare(self, model, target_class, target_percentage, num_epochs):
        rng = np.random.RandomState(seed=9112018)
        train_data = data_providers.MNISTDataProvider('train', batch_size=50, rng=rng, max_num_batches=100)
        valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng, max_num_batches=100)

        optimizer = optim.SGD(model.parameters(), lr=1e-1)

        # get new inputs/targets
        m = ModifyDataProvider()
        inputs_full, targets_full, inputs_red, targets_red = m.modify(target_class, target_percentage, train_data.inputs, train_data.targets)
        inputs_full_valid, targets_full_valid, inputs_red_valid, targets_red_valid = m.modify(target_class, target_percentage, valid_data.inputs, valid_data.targets)

        # print(m.get_label_distribution(targets_full, 'full'))
        # print(m.get_label_distribution(targets_red, 'reduced'))

        # train full
        train_data.inputs = np.array(inputs_full)
        train_data.targets = np.array(targets_full)
        valid_data.inputs = np.array(inputs_full_valid)
        valid_data.targets = np.array(targets_full_valid)
        # m.get_label_distribution(targets_full_valid, 'full')
        # m.get_label_distribution(targets_red_valid, 'reduced')

        model = SimpleFNN(input_shape=(28, 28), h_out=100, num_classes=10)
        train_acc_full, train_loss_full = self._train(model, 'full_data_test', train_data, num_epochs, optimizer)
        valid_acc_full, valid_loss_full = self._evaluate(model, 'full_data_test', valid_data, [i for i in range(num_epochs)])

        # train reduced
        train_data.inputs = np.array(inputs_red)
        train_data.targets = np.array(targets_red)
        valid_data.inputs = np.array(inputs_red_valid)
        valid_data.targets = np.array(targets_red_valid)
        model = SimpleFNN(input_shape=(28, 28), h_out=100, num_classes=10)
        optimizer = optim.SGD(model.parameters(), lr=1e-1)

        print(len(train_data.inputs), len(train_data.targets))

        train_acc_red, train_loss_red = self._train(model, 'reduced_data_test', train_data, num_epochs, optimizer)

        # valid_acc_red, valid_loss_red = self._evaluate(model, 'reduced_data_test', valid_data, [i for i in range(num_epochs)])

        # calculate differences
        train_acc_diff = ((train_acc_full - train_acc_red) / float(train_acc_full)) * 100
        train_loss_diff = ((train_loss_full - train_loss_red) / float(train_loss_full)) * 100
        valid_acc_diff = ((valid_acc_full - valid_acc_red) / float(valid_acc_full)) * 100
        valid_loss_diff = ((valid_loss_full - valid_loss_red) / float(valid_loss_full)) * 100

        return train_acc_diff, train_loss_diff, valid_acc_diff, valid_loss_diff

    def play(self, target_class, target_percentage):
        # setup hyperparameters
        num_epochs = 100
        model = SimpleFNN(input_shape=(28, 28), h_out=100, num_classes=10)
        train_acc_diff, train_loss_diff, valid_acc_diff, valid_loss_diff = self._compare(model, target_class, target_percentage, num_epochs)
        return atrain_acc_diff, train_loss_diff, valid_acc_diff, valid_loss_diff


def driver():
    data = {}
    target_percentage = .01
    print("Setting percentage reduction to " + str(target_percentage))
    for i in range(0, 10):
        train_acc_diff, train_loss_diff, valid_acc_diff, valid_loss_diff = Experiment().play(i, target_percentage)
        data[i] = {"Target Percentage (in %)": target_percentage * 100, "Label": i, "Train_Acc_Diff": train_acc_diff, "Train_Loss_Diff": train_loss_diff, "Valid_Acc_Diff": valid_acc_diff, "Valid_Loss_Diff": valid_loss_diff}

    with open('data/minority_classes_output.csv', 'w') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, values in data.items():
            writer.writerow(values)


driver()
