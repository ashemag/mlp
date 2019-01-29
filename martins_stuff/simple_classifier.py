import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from martins_stuff import data_providers
from martins_stuff.experiment_builder import Network


class SimpleNeuralClassifier(Network):
    def __init__(self, input_shape, h_out, num_classes):
        super(SimpleNeuralClassifier,self).__init__()
        self.input_shape = input_shape

        in_features = input_shape[0]*input_shape[1] # height*width
        self.hidden_layer = nn.Linear(in_features, h_out, bias=False)
        self.output_layer = nn.Linear(h_out,num_classes,bias=False)

    def forward(self, x):
        """
        :param x: tensor.
        :return:
        """
        pred = x
        pred = self.hidden_layer(pred)
        pred = torch.sigmoid(pred)
        pred = self.output_layer(pred)

        # note: output not softmax because CrossEntropyLoss in pytorch does the softmax transform for you

        return pred

def test_load_model():
    '''
    example of how to load a model. model that is loaded is a simple neural network i trained for 50 epochs.
    at each epoch model is saved in folder .../saved_models/test_simple/saved_models_train. model at 40-th
    epoch is loaded.
    :return:
    '''

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    saved_models_dir = os.path.join(ROOT_DIR, 'martins_stuff/saved_models')
    model_path = os.path.join(saved_models_dir,'saved_models_train/model_40')
    input_shape = (28, 28)
    num_classes = 10

    # in order to load model you need to create a model that has same structure as the one you saved
    model = SimpleNeuralClassifier(input_shape, h_out=100, num_classes=num_classes)
    model.load_model(model_path)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Test accuracy.

    seed = 9112018; rng = np.random.RandomState(seed=seed)
    train_data = data_providers.MNISTDataProvider('train',batch_size=100, rng=rng,max_num_batches=100) # is an iterator.

    x_batch, y_batch = train_data.next() # arrays size (batch_size,-1)
    acc = model.get_acc_batch(x_batch,y_batch) # calc. accuracy on given batch

    print("accuracy of model on batch: ",acc)

def test_train_and_save():
    '''
    example of training a model and saving its results.
    :return:
    '''
    seed = 9112018
    rng = np.random.RandomState(seed=seed)
    train_data = data_providers.MNISTDataProvider('train',batch_size=100, rng=rng,max_num_batches=100)

    input_shape = (28, 28)
    num_classes = 10
    num_epochs = 50

    model = SimpleNeuralClassifier(input_shape, h_out=100, num_classes=num_classes)
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    saved_models_dir = os.path.join(ROOT_DIR, 'martins_stuff/saved_models')
    train_results_file = os.path.join(ROOT_DIR,'martins_stuff/experiments/training_results_simple.txt')

    model.train_full(train_data, num_epochs, optimizer, train_results_file, saved_models_dir)

def main():
    # test_load_model()
    pass

if __name__ == '__main__':
    main()