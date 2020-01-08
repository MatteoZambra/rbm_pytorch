
import os
import sys
sys.path.append(os.getcwd() + r'\img_util')

import images_utils as iu
import rbm_utils as ru

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms



#%% Data import and visualization

batch_size = 300

transforms = transforms.Compose([transforms.ToTensor()])

train_data = MNIST('data/', download = True, train = True, transform = transforms)
test_data  = MNIST('data/', download = True, train = False, transform = transforms)

train_load = DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_load  = DataLoader(test_data, batch_size = batch_size, shuffle = True)

data_iter = iter(train_load)
images, labels = data_iter.next()

iu.images_plot(images, labels)
images = iu.binarize_digits(images, factor = 3.5)
iu.images_plot(images, labels)


#%% instantiation of the RBM

images = images.view(-1, images.shape[2] * images.shape[3])
visible_dim = images.shape[1]
hidden_dim = 100
epochs = 20
mcmc_steps = 50
learning_rate = 0.0001
momentum = 0.8
weight_decay = 0.00001

rbm = ru.RBM(visible_dim, hidden_dim)
rbm.train(train_load, epochs, learning_rate, weight_decay, momentum, mcmc_steps)


#%%

data_iter = iter(test_load)

for i in range(3):
    data,labels = data_iter.next()
    samples = data.view(-1, data.shape[2] * data.shape[3])
    samples = iu.binarize_digits(samples, factor = 3.5)
    rbm.generate_samples(samples,labels)
#end










