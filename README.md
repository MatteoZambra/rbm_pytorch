# Restricted Boltzmann Machine Walkthrough. PyTorch implementation


**NOTE** This walkthrough is heavily inspired by [this tutorial](https://heartbeat.fritz.ai/guide-to-restricted-boltzmann-machines-using-pytorch-ee50d1ed21a8), a very clear introductory guide to Restricted Boltzmann Machines in PyTorch by Dr. Derrick Mwiti. There a different data set is used, however. I modified the first stages of data loading an preprocessing so that the same device can work on a MNIST data set. A more comprehensive and broader account on this latter is given in [this other tutorial](https://github.com/iam-mhaseeb/Multi-Layer-Perceptron-MNIST-with-PyTorch/blob/master/mnist_mlp_exercise.ipynb), in which a Feedforward MLP is built and trained on the MNIST data set. Note also that the data set is loaded with the `torch.utils.data.DataLoader` utility method provided by PyTorch.

The files `images_utils.py` and `rmb_utils.py` serve the purposes to provide functions to manipulate images and train the device itself, respectively. 

Note that image manipulation is necessary, inasmuch the MNIST data set samples come as real-valued matrices but the RMB as here implemented can only deal with **binary** variables, that is values among $\{0, +1\}$. A data sample is binarized according to a user-defined threshold. 

## Companion files overview
### `images_utils.py`

Contains plotting utilities, such as visualizing data samples, images binarization, model parameters histograms, receptive  fields visualization. These latter two graphical utilities are recommended in [this RBMs training tutorial](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) (Hinton, 2010).

Histograms serve the purpose to monitor the distribution of model parameters and their variations. Learning rate should be set in such a way to render the weights updates magnitudes about 0.001 times the weights themselves. Moreover, observe that in RBMs the hidden units act as *feature detectors* hence the receptive fields visualizations may help to understand which are the features that make the hidden neurons fire.

### `rbm_utils.py`

Contains the `rbm` class, containing all the training and parameters updating utility methods.

**NOTE** that the program is written with PyTorch, hence the data samples are `torch.Tensor` types. The useful functions have these data types as input mainly.

## Remarks 

As pointed out in [Hinton (2010)](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf), in the last step of the Gibbs chain which is run to obtain unbiased estimated of the log-likelihhod gradients, hidden activity patterns should be sampled as probabilities, instead of binarizing such probabilities to obtain a vector of binary values as done for the other sampling. This reduces sampling noise. 

## Further improvements
A RBM is a single hidden layer model. However the power of Deep Learning stems from the higher level of abstraction that can be gained with a *deep* architecture. While the results of this simulation may seem satisfactory, since some digits are correctly rencostructed, better results may be attained with deeper architectures.

What can be improved may be

* A deeper network. It would be a different model, namely *Deep Belief Network*. Note that it does not suffice to make a RBM deeper for the sake of building a *Deep Boltzmann Machine*. While being structurally similar, DBNs and DBMs differ since the former are directed Probabilistic Graphical Models
* binarize all the samples in all the batches before training, instead of binarizing a data batch once it is fetched from the dedicated data structure in the training loop
* Reconstruction error (MSE), used to monitor learning, does not provide a sensible measure of how well the device is learning, since it does not descend naturally from the objective function (Contrastive Divegence) that is minimized. The *free energy* of training samples and held-out samples could rather be a neater measure. See [Zorzi et al. (2013)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3747356/)
* Enforce sparsity
