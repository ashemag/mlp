import torch.nn as nn
import torch
from torchvision.models.densenet import DenseNet
from torchvision.models.densenet import densenet161
from ModelBuilder.base import Network
import numpy as np
import data_providers
from collections import OrderedDict

from torchvision.models.densenet import _DenseBlock
from torchvision.models.densenet import _Transition

class DenseNetWrapper(DenseNet,Network):
    def __init__(self,in_channels=3,growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        Network.__init__(self)
        DenseNet.__init__(self,
                          num_init_features=num_init_features,
                          growth_rate=growth_rate,
                          block_config=block_config)

        self.features.conv0 = nn.Conv2d(in_channels,out_channels=num_init_features, kernel_size=7, stride=2, padding=3, bias=False)

        # TODO: Adapt num_classes to be custome (e.g. 10 not 1000)

class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()


        num_init_features = 64
        in_channels = 1
        self.conv0 = nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers_dict = {}
        block_config = (6, 12, 24, 16)
        bn_size = 4
        growth_rate = 32
        drop_rate = 0

        num_features = num_init_features
        for i, num_layers in enumerate(block_config): # 0,1,2,3
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.layers_dict['denseblock_{}'.format(i+1)] = block
            num_features = num_features + num_layers * growth_rate

            if i!= len(block_config)-1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.layers_dict['transition_{}'.format(i+1)] = trans
                num_features = num_features // 2

        self.layers_dict['norm5'] = nn.BatchNorm2d(num_features)



    def forward(self,x):

        pred = self.conv0(x)
        pred = self.norm0(pred)
        pred = self.relu0(pred)
        pred = self.pool0(pred)

        print(self.layers_dict.keys())

        print("pool0: ",pred.size())

        pred = self.layers_dict['denseblock_1'](pred)
        print("dense1: ", pred.size())
        pred = self.layers_dict['transition_1'](pred)
        print("trans1 ", pred.size())
        pred = self.layers_dict['denseblock_2'](pred)
        print("dense2: ", pred.size())
        pred = self.layers_dict['transition_2'](pred)
        print("trans2: ", pred.size())
        pred = self.layers_dict['denseblock_3'](pred)
        # pred = self.layers_dict['transition_3'](pred)
        # pred = self.layers_dict['denseblock_4'](pred)
        #pred = self.layers_dict['transition_1'](pred)

        # Each denseblock
        num_init_features = 64
        num_features = num_init_features






        return pred




def test_forward_pass():
    from mlp import data_providers
    import numpy as np
    import data_providers

    rng = np.random.RandomState(seed=9112018)
    valid_data = data_providers.MNISTDataProvider('valid', batch_size=100, rng=rng, max_num_batches=100)
    x, y = valid_data.next()

    x = np.reshape(x,(100,1,28,28))
    print("x shape: {}, x type: {}, y shape: {}, y type: {}".format(x.shape,y.shape,type(x),type(y)))

    x_tens = torch.Tensor(x).float()
    model = DenseNetWrapper(in_channels=1)
    model_test = TestNetwork()
    pred = model_test(x_tens)

    #pred = model(x_tens)

    print(pred.size())

    print("success")


    pass

def test_forward2():
    num_init_features = 64
    features = nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
    ]))
    print(features)
    features.conv0 = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
    print(features)
    # print(features)


    pass

def main():
    test_forward_pass()
    # test_forward2()

    pass

if __name__ == '__main__':
    main()