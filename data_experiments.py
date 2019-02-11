from mlp_resources.data_providers import *
from ModelBuilder.simple_fnn import *
import numpy as np
import globals
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torchvision import transforms, datasets

os.environ['MLP_DATA_DIR'] = 'data'


class Experiment(object):
    @staticmethod
    def _train(model, model_title, train_data, num_epochs, optimizer):
        saved_models_dir = os.path.join(globals.ROOT_DIR, 'SavedModels/' + model_title)
        train_results_file = os.path.join(globals.ROOT_DIR, 'ExperimentResults/' + model_title + '.txt')
        model.train_full(train_data, num_epochs, optimizer, train_results_file, saved_models_dir)
        with open('ExperimentResults/' + model_title + '.txt') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t')
            for row in csv_reader:
                train_acc, train_loss = float(row['train_acc']), float(row['train_loss'])
        return train_acc, train_loss

    @staticmethod
    def _evaluate(model, model_title, valid_data, num_epochs):
        saved_models_dir = os.path.join(globals.ROOT_DIR, 'SavedModels/' + model_title)
        eval_results_file = os.path.join(globals.ROOT_DIR, 'ExperimentResults/' + model_title + '_eval.txt')
        model.evaluate_full(valid_data, num_epochs, saved_models_dir, eval_results_file)

        with open('ExperimentResults/' + model_title + '.txt') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t')
            for row in csv_reader:
                eval_acc, eval_loss = float(row['train_acc']), float(row['train_loss']) #bug in code
        return eval_acc, eval_loss

    @staticmethod
    def _process_EMNIST(inputs):
        samples, _, height, width = inputs.shape
        return inputs.reshape(samples, height * width)

    def _compare(self, train_data_full, train_data_mod, test_data):
        rng = np.random.RandomState(seed=9112018)
        num_epochs = 100

        # TRAIN FULL
        model_full = SimpleFNN(input_shape=(28, 28), h_out=100, num_classes=10)
        optimizer = optim.SGD(model_full.parameters(), lr=1e-1)
        train_acc_full, train_loss_full = self._train(model_full, 'full_data_test', train_data_full, num_epochs, optimizer)
        valid_acc_full, valid_loss_full = self._evaluate(model_full, 'full_data_test', test_data,
                                                         [i for i in range(num_epochs)])

        # TRAIN REDUCED
        model_mod = SimpleFNN(input_shape=(28, 28), h_out=100, num_classes=10)
        optimizer = optim.SGD(model_full.parameters(), lr=1e-1)
        train_acc_mod, train_loss_mod = self._train(model_mod, 'full_data_test', train_data_mod, num_epochs, optimizer)
        valid_acc_mod, valid_loss_mod = self._evaluate(model_mod, 'full_data_test', test_data,
                                                         [i for i in range(num_epochs)])

        # calculate differences
        train_acc_diff = ((train_acc_mod - train_acc_full) / float(train_acc_mod)) * 100
        train_loss_diff = ((train_loss_mod - train_loss_full) / float(train_loss_mod)) * 100
        valid_acc_diff = ((valid_acc_mod - valid_acc_full) / float(train_acc_mod)) * 100
        valid_loss_diff = ((valid_loss_mod - valid_loss_full) / float(valid_loss_mod)) * 100

        return train_acc_diff, train_loss_diff, valid_acc_diff, valid_loss_diff


def driver():
    data = {}
    target_percentage = .005
    print("Setting percentage reduction to " + str(target_percentage))
    for i in range(0, 10):
        print("ON LABEL {0}".format(i))
        train_acc_diff, train_loss_diff, valid_acc_diff, valid_loss_diff = Experiment().play(i, target_percentage)
        data[i] = {"Target Percentage (in %)": target_percentage * 100, "Label": i, "Train_Acc_Diff (%)": train_acc_diff, "Train_Loss_Diff (%)": train_loss_diff, "Valid_Acc_Diff (%)": valid_acc_diff, "Valid_Loss_Diff (%)": valid_loss_diff}
        break
    with open('data/minority_classes_output.csv', 'w') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, values in data.items():
            writer.writerow(values)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def cifar_driver():
    d = unpickle('data/cifar-10-batches-py/batches.meta')
    labels = d[b'label_names']
    m = ModifyDataProvider()
    train_set = CIFAR10(root='data', set_name='train')
    inputs = [i[0] for i in train_set]
    labels = [labels[i[1]] for i in train_set]
    m.get_label_distribution(labels)

    target_percentage = .01
    label = b'horse'
    inputs_full, targets_full, inputs_mod, targets_mod = m.modify(label, target_percentage, inputs, labels)
    m.get_label_distribution(targets_full)

    # get test data
    test_set = CIFAR10(root='data', set_name='test')
    m.get_label_distribution([labels[i[1]] for i in test_set], "Test Set Full")

    train_set_full = zip(inputs_full, targets_full)
    train_set_mod = zip(inputs_mod, targets_mod)

    train_acc_diff, train_loss_diff, valid_acc_diff, valid_loss_diff = Experiment()._compare(train_set_full, train_set_mod, test_set)
    output = {"Target Percentage (in %)": target_percentage * 100, "Label": label,
              "Train_Acc_Diff (%)": train_acc_diff, "Train_Loss_Diff (%)": train_loss_diff,
              "Valid_Acc_Diff (%)": valid_acc_diff, "Valid_Loss_Diff (%)": valid_loss_diff}
    print(output)


# driver
cifar_driver()
