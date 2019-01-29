from martins_stuff.simple_classifier import SimpleNeuralClassifier
from martins_stuff import data_providers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def cross_entropy_loss_obs(x_obs, y_obs_target, model):
    '''
    :param: x_obs is a numpy array of size (1,-1). represents a single observation of an image.
    :param: y_obs_target numpy array of size (1,-1). is one-hot-encoded. gives true label of x_obs
    :return: adversarial example. is a numpy array of size (1,-1)
    '''

    # F.cross_entropy requires target to be integer encoded
    y_obs_desired_int = np.argmax(y_obs_target, axis=1)
    y_obs_desired_int = torch.Tensor(y_obs_desired_int).long()  # notice brackets []

    # input to model must be tensor of type float
    x_obs_temp = torch.Tensor(x_obs).float()
    x_obs_temp.requires_grad = True
    y_obs_pred = model(x_obs_temp)  # returns tensor shape (-1,) of predicted class probabilities
    y_obs_pred = torch.reshape(y_obs_pred, (1, -1))  # required shape for cross_entropy
    loss = F.cross_entropy(input=y_obs_pred, target=y_obs_desired_int)

    return loss, x_obs_temp

def targeted_fast_gradient_sign_attack(x_obs,y_obs_desired,model,epsilon=100):
    '''
    instead of moving in direction that worsens the loss we try to improve loss wrt. to desired target.
    source: https://medium.com/onfido-tech/adversarial-attacks-and-defences-for-convolutional-neural-networks-66915ece52e7
    :return:
    '''

    loss, x_obs_temp = cross_entropy_loss_obs(x_obs,y_obs_desired,model)
    loss.backward() # calculates (does not update) gradient delta_loss/delta_x for ever x that has requires_grad = true
    grad_wrt_x_obs = x_obs_temp.grad.numpy()  # returns array of size (-1,)
    grad_wrt_x_obs = np.reshape(grad_wrt_x_obs, (1, -1))  # row vector format (same format as input)
    advers_x_obs = x_obs - epsilon * np.sign(grad_wrt_x_obs)

    return advers_x_obs

def fast_gradient_sign_attack(x_obs, y_obs_true, model, epsilon=100):
    '''
    gradient wrt input x shows how much the loss change wrt to small change in x. this fact is
    used to construct an example that worsens the loss of an observation.

    :param: x_obs is a numpy array of size (1,-1). represents a single observation of an image.
    :param: y_obs_true numpy array of size (1,-1). is one-hot-encoded. gives true label of x_obs
    :return: adversarial example. is a numpy array of size (1,-1)
    '''

    loss, x_obs_temp = cross_entropy_loss_obs(x_obs,y_obs_true,model)
    loss.backward() # calculates (does not update) gradient delta_loss/delta_x for ever x that has requires_grad = true
    grad_wrt_x_obs = x_obs_temp.grad.data.numpy() # returns array of size (-1,)
    grad_wrt_x_obs = np.reshape(grad_wrt_x_obs,(1,-1)) # row vector format (same format as input)
    advers_x_obs = x_obs + epsilon*np.sign(grad_wrt_x_obs)

    return advers_x_obs


def get_fooling_targets(true_target_int,num_classes):
    # returns one-hot-encoded fooling targets (targets not equal to true_target)

    fooling_classes = []  # all integers except x_batch[0] (7)
    for k in range(num_classes):
        if k != true_target_int[0]:
            fooling_classes.append(k)

    # next he one-hot-encodes them

    foolingtargets = np.zeros((len(fooling_classes), num_classes))
    for n in range(len(fooling_classes)):
        foolingtargets[n, fooling_classes[n]] = 1

    return foolingtargets

def plot_things(plot_dict,epsilon):
    '''
    :param x_batch: array size (batch_size,-1)
    :return:
    '''

    x_advers_batch = plot_dict['x_advers_batch']
    y_desired_ints = plot_dict['desired_targets']
    y_desired_probs = plot_dict['desired_targets_prob']
    y_predicted_ints = plot_dict['predicted_targets']
    y_predicted_probs = plot_dict['predicted_targets_prob']
    # {:.4f}
    plt.figure()

    for i in range(len(x_advers_batch)):
        plt.subplot(3,3,i+1)
        plt.imshow(x_advers_batch[i].reshape((28,28)),cmap='Greens')
        title = "desired: {} prob: {:.3f}".format(y_desired_ints[i],y_desired_probs[i])
        xlabel = "predicted: {} prob: {:.3f}".format(y_predicted_ints[i],y_predicted_probs[i])
        plt.title(title)
        plt.xlabel(xlabel)

    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("epsilon: {0}".format(epsilon))
    plt.show()

def main():
    model_path = "C:/test_simple/saved_models_train/model_40"

    input_shape = (28, 28)
    num_classes = 10
    model = SimpleNeuralClassifier(input_shape, h_out=100, num_classes=num_classes)
    model.load_model(model_path) # pre-trained model (acc 85%)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    test_data = data_providers.MNISTDataProvider('test', batch_size=100, rng=None,max_num_batches=1,shuffle_order=False)

    x_batch, y_batch = test_data.next()
    y_batch_int = np.argmax(y_batch,axis=1)

    y_obs = np.reshape(y_batch_int[0],(1,1))
    x_obs = np.reshape(x_batch[0],(1,-1)) # numpy row vector

    y_targets = get_fooling_targets(y_obs,num_classes=10) # one-hot-encoded

    plot_dict = {}
    plot_dict['desired_targets'] = []
    plot_dict['desired_targets_prob'] = []
    plot_dict['predicted_targets'] = []
    plot_dict['predicted_targets_prob'] = []
    plot_dict['x_advers_batch'] = []

    eps = 0.15

    for y_target in y_targets:
        # get pred of advers example, and accuracy:
        y_target = np.reshape(y_target,(1,-1))
        x_advers = targeted_fast_gradient_sign_attack(x_obs, y_target, model, epsilon=eps) # array (1,-1)
        x_advers_tens = torch.Tensor(x_advers).float()
        y_advers_pred_tens = F.softmax(model(x_advers_tens))
        y_advers_pred = y_advers_pred_tens.data.numpy() # array (1,-1)

        desired_target = np.argmax(y_target)
        plot_dict['desired_targets'].append(desired_target)

        print(y_advers_pred," sum: ",np.sum(y_advers_pred))
        plot_dict['desired_targets_prob'].append(y_advers_pred[0,desired_target])
        plot_dict['predicted_targets'].append(np.argmax(y_advers_pred))
        plot_dict['predicted_targets_prob'].append(np.max(y_advers_pred))
        plot_dict['x_advers_batch'].append(x_advers)

    temp = plot_dict['x_advers_batch']
    plot_dict['x_advers_batch'] = np.array(temp).reshape(len(temp),-1)

    plot_things(plot_dict,epsilon=eps)

if __name__ == '__main__':
    main()