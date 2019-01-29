from mlp.data_providers import MNISTDataProvider, ModifyDataProvider
from mlp.models import MultipleLayerModel
from mlp.layers import ReluLayer, AffineLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.initialisers import GlorotUniformInit, ConstantInit
from mlp.learning_rules import MomentumLearningRule
from mlp.optimisers import Optimiser
import numpy as np
import os
os.environ['MLP_DATA_DIR'] = 'data'


class Experiment(object):
    def _train(self, model, num_epochs, train_data, valid_data):
        # Multiclass classification therefore use cross-entropy + softmax error
        error = CrossEntropySoftmaxError()

        # Use a momentum learning rule - you could use an adaptive learning rule
        # implemented for the coursework here instead
        learning_rule = MomentumLearningRule(0.02, 0.9)
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

    def _compare(self, model, num_epochs):
        train_data = MNISTDataProvider('train')
        valid_data = MNISTDataProvider('valid')

        err_train, err_valid, acc_train, acc_valid = self._train(model, num_epochs, train_data, valid_data)

        inputs, targets = ModifyDataProvider().modify(0, .01, train_data.inputs, train_data.targets)
        train_data.inputs = np.array(inputs)
        train_data.targets = np.array(targets)
        err_train_comp, err_valid_comp, acc_train_comp, acc_valid_comp = self._train(model, num_epochs, train_data, valid_data)

        acc_valid_diff = ((acc_valid_comp - acc_valid) / float(acc_valid_comp)) * 100
        acc_train_diff = ((acc_train_comp - acc_train) / float(acc_train_comp)) * 100
        print("\n===\n")
        print("Training accuracy decreased by {0}%".format(round(acc_train_diff, 2)))
        print("Validation accuracy decreased by {0}%".format(round(acc_valid_diff, 2)))
        print("\n===\n")

    def play(self):
        # Seed a random number generator
        seed = 31102016
        rng = np.random.RandomState(seed)

        # Probability of input being included in output in dropout layer
        incl_prob = 0.5
        input_dim, output_dim, hidden_dim = 784, 10, 125

        # Use Glorot initialisation scheme for weights and zero biases
        weights_init = GlorotUniformInit(rng=rng, gain=2. ** 0.5)
        biases_init = ConstantInit(0.)

        # Create three affine layer model with rectified linear non-linearities
        # and dropout layers before every affine layer
        model = MultipleLayerModel([
            AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
            ReluLayer(),
            AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
        ])
        self._compare(model, 2)



Experiment().play()

