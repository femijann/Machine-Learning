import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis = 0
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    c = X.shape[1]       # number of columns of the array
    for j in range(c):
      hypothesis = X[i, j] * theta[j] + hypothesis
    ########################################/
    
    return hypothesis
