
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')


def binarize_digits(X, factor):
    """
    An image sample is a numerical matrix
    Since the MNIST data set is loaded with the torch.utils.data method,
    then the data set samples are torch.Tensor types
    
    Input:
        ~ X:      a samples batch
        ~ factor: float
        
    Returns:
        ~ X: processed input batch
        
    Note that these images
    """
    
    max_value = torch.max(X).numpy()
    threshold = max_value / factor
    
    X[X <= threshold] = 0.
    X[X >  threshold] = 1.
    
    return X
#end
    

def images_plot(X,Y):
    """
    Plot a grid of, say, 2x5 images, regardless of whether they are binarized 
    or not.
    
    Input:
        ~ X: torch.Tensor batch of samples. Regardless of the batch size,
             10 digits are being used
        ~ Y: torch.Tensor, respective labels of X
        
    Returns:
        nothing
    """
    
    X = X.numpy()
    X = np.squeeze(X)
    Y = Y.numpy()

    fig = plt.figure(figsize=(15,4))
    for idx in range(10):
        ax = fig.add_subplot(2,10/2, idx+1, xticks=[], yticks=[])
        ax.imshow(X[idx], cmap = 'gray')
        ax.set_title(Y[idx])
    #end
        
    plt.show()
#end

def plot_digit(image):
    """
    Plot the sample image as numerical matrix, with respective pixel values
    """
    
    if (type(image) is not np.ndarray):
        image = image.numpy()
        image = np.squeeze(image)
    #end
    width, height = image.shape
    threshold = max(image.flatten())/2.5
    
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap = 'gray')

    for x in range(width):
        for y in range(height):
            value = round(image[x][y], 2) if image[x][y] != 0 else 0
            #value = round(image[x][y], 0) if image[x][y] > threshold else 0
            ax.annotate(str(value), xy = (y,x),
                        horizontalalignment = 'center',
                        verticalalignment   = 'center',
                        color = 'white' if image[x][y] < threshold else 'black')
        #end
    #end
    
#end

    
def plot_params_histogram_(w, dw, a, da, b, db):
    """
    Written for the purpose of plotting the model parameters values histrograms
    
    Input:
        ~ w, dw: connections strengths and their variations
        ~ a, da: visile units biases and their variations
        ~ b, db: hidden units biases and their variations
        
        All these are torch.Tensor data types
    
    Returns:
        nothing
    """
    
    fig = plt.figure(figsize = (14,6.5))
    ax = fig.add_subplot(231)
    ax.hist(w.reshape(1, w.shape[0] * w.shape[1]))
    ax.set_title("Weights")
    ax = fig.add_subplot(232)
    ax.hist(dw.reshape(1, dw.shape[0] * dw.shape[1]))
    ax.set_title("Weights Variations")
    ax = fig.add_subplot(233)
    ax.hist(a)
    ax.set_title("Visible Bias")
    ax = fig.add_subplot(234)
    ax.hist(da)
    ax.set_title("Visible Bias Variation")
    ax = fig.add_subplot(235)
    ax.hist(b)
    ax.set_title("Hidden Bias")
    ax = fig.add_subplot(236)
    ax.hist(db)
    ax.set_title("Hidden Bias Variation")
    
    plt.show()
    plt.close('all')
#end


def receptive_fields_plot(W, a ,b):
    """
    Plots the gray-scale values of the model parameters.
    
    Input:
        ~ W:   connection strengths matrix
        ~ a,b: visible and hidden units biases
        
    Returns:
        nothing
    """
    
#    fig,ax = plt.subplots(3, figsize=(10,10))
#    title_dict = {0 : 'Weights',
#                  1 : 'Visible bias',
#                  2 : 'Hidden bias'}
#    
#    for i,param in zip(range(3), [W,a,b]):
#        ax[i].imshow(param, cmap = 'gray')
#        ax[i].xaxis.set_ticks_position('none')
#        ax[i].yaxis.set_ticks_position('none')
#        ax[i].set_title(title_dict[i])
#    #end
    
    side_dim = int(np.sqrt(W.shape[0]))
    hidden_dim = int(np.sqrt(W.shape[1]))
    
    fig = plt.figure(figsize=(15,15))
    for i in range(W.shape[0]):
        ax = fig.add_subplot(side_dim, side_dim, i+1, xticks = [], yticks = [])
        ax.imshow(W[i,:].view(hidden_dim, hidden_dim), cmap = 'gray')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
    #end
        
    plt.show()
#end


def cost_profile_plot(cost):
    """
    Plot the trend of the MSE error metric.
    Recall, ``Use it but don't trust it'' (Hinton, 2010)
    MSE is not the objective function that is minimized hence is not
    a fully reliable performance metric
    """
    
    fig = plt.figure(figsize=(10,5))
    ax = plt.gca()
    ax.plot(np.arange(1,len(cost) + 1), cost, c = 'r', alpha = 0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Error (MSE)')
    plt.show()
#end




























    
    