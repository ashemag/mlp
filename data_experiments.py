from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider, ModifyDataProvider
from mlp.models import MultipleLayerModel
from mlp.layers import ReluLayer, AffineLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.initialisers import GlorotUniformInit, ConstantInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser
import numpy as np
import os
import torch
os.environ['MLP_DATA_DIR'] = 'data'


class Experiment(object):
    def _train(self, model, num_epochs, train_data, valid_data):
        # Multiclass classification therefore use cross-entropy + softmax error
        error = CrossEntropySoftmaxError()

        # Use a momentum learning rule - you could use an adaptive learning rule
        # implemented for the coursework here instead
        learning_rule = AdamLearningRule()
        # Monitor classification accuracy during training
        data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

        optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)
        stats_interval = 1
        stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)
        err_train = stats[len(stats) - 1, keys['error(train)']]
        err_valid = stats[len(stats) - 1, keys['error(valid)']]
        acc_train = stats[len(stats) - 1, keys['acc(train)']]
        acc_valid = stats[len(stats) - 1, keys['error(valid)']]
        return err_train, err_valid, acc_train, acc_valid

    def _compare(self, model, target_class, target_percentage, num_epochs):
        train_data = MNISTDataProvider('train', batch_size=100)
        valid_data = MNISTDataProvider('test', batch_size=100)

        err_train, err_valid, acc_train, acc_valid = self._train(model, num_epochs, train_data, valid_data)

        m = ModifyDataProvider()
        m.get_label_distribution(train_data.targets)
        inputs, targets = m.modify(target_class, target_percentage, train_data.inputs, train_data.targets)
        train_data.inputs = np.array(inputs)
        train_data.targets = np.array(targets)
        m.get_label_distribution(targets)

        err_train_comp, err_valid_comp, acc_train_comp, acc_valid_comp = self._train(model, num_epochs, train_data, valid_data)

        acc_valid_diff = ((acc_valid_comp - acc_valid) / float(acc_valid_comp)) * 100
        acc_train_diff = ((acc_train_comp - acc_train) / float(acc_train_comp)) * 100
        print("\n===\n")
        print("Training accuracy changed by {0}%".format(round(acc_train_diff, 2)))
        print("Validation accuracy changed by {0}%".format(round(acc_valid_diff, 2)))
        print("\n===\n")

    def play(self, target_class, target_percentage):
        # Seed a random number generator
        seed = 31102016
        rng = np.random.RandomState(seed)

        # setup hyperparameters
        num_epochs = 100
        input_dim, output_dim, hidden_dim = 784, 10, 100

        weights_init = GlorotUniformInit(rng=rng)
        biases_init = ConstantInit(0.)
        # model = MultipleLayerModel([
        #     AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
        #     ReluLayer(),
        #     AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        #     ReluLayer(),
        #     AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
        # ])
        model = torch.nn.Linear(input_dim, output_dim)
        self._compare(model.parameters(), target_class, target_percentage, num_epochs)


Experiment().play(0, .01)

