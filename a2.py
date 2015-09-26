# a2.py - 
# by K. Brett Mulligan
# CSU - CS545 (Fall 2015)
# Assignment 2
# This assignment explores ridge regression using the UCI winequality dataset

from __future__ import division

import math
import time
import numpy as np
from matplotlib import pyplot as plt

red_file = "winequality-red.csv"
white_file = "winequality-white.csv"


class RidgeRegression :

    def __init__(self):
        pass

    def fit(self, d):
        """Given dataset 'd', calculate the ridge regression using either RMSE or MAD."""
        return


class Dataset :

    def __init__(self):
        pass

class WineDataset(Dataset):

    def __init__(self, filename):
        self.winedata = np.genfromtxt(filename, delimiter=';')[1:]       # remove the first line (featureset description)

    def standardize(self, data):
        """Given matrix data, standardizes each feature by subtracting mean and
        dividing by standard deviation. Returns new matrix of same shape with 
        standardized values."""

        new_data = np.copy(np.transpose(data))

        for i in range(len(new_data)):
            self.mean = np.mean(new_data[i])
            self.std_dev = np.std(new_data[i])
            new_data[i] = [(x - self.mean)/self.std_dev for x in new_data[i]]

        return np.transpose(new_data)

    def unstandardize_value(self, val):
        return val * self.std_dev + self.mean




def analyze(data, plot_this=False):
    print data
    print 'Length:', len(data)
    print 'Features:', len(data[0])

    scores = [x[11] for x in data]

    print 'Max:', max(scores)
    print 'Min:', min(scores)

    if (plot_this == True):
        plt.hist(scores)
        plt.show()

def mad(w, data, labels):

    # iterate over all examples

    total = 0
    N = len(data)
    for i in range(N):
        error = labels[i] - predict(w, data[i])
        total += np.absolute(error)

    return total/float(N)

def rmse(w, data, labels):

    # iterate over all examples

    total = 0
    N = len(data)
    for i in range(N):
        error = labels[i] - predict(w, data[i])
        total += np.square(error)

    return np.sqrt(total/float(N))

def standardize(data):
    """Given matrix data, standardizes each feature by subtracting mean and
    dividing by standard deviation. Returns new matrix of same shape with 
    standardized values."""

    new_data = np.copy(np.transpose(data))

    for i in range(len(new_data)):
        mean = np.mean(new_data[i])
        std_dev = np.std(new_data[i])
        new_data[i] = [(x - mean)/std_dev for x in new_data[i]]

    return np.transpose(new_data)

def extract(data):
    """Given data in the form of examples in rows terminated by label, strips all labels.
    Returns unlabeled data and labels in separate matrices."""
    examples = data.T[:-1].T
    labels = data.T[-1]
    return examples, labels

def regress(X, y):
    """Returns weight vector w by solving system of equations in 
    matrix form. Requires training data X and labels y."""
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    return w

def regress_slow(X, y):
    """Returns weight vector w by solving directly for w. 
    Requires training data X and labels y."""
    w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return w

def regress_ridge(X, y, reg):
    """Returns weight vector w by solving system of equations in 
    matrix form. Requires regularization term reg, training data X and labels y."""
    product = np.dot(X.T, X)
    w = np.linalg.solve(product + reg * np.identity(len(product)), np.dot(X.T, y))
    return w

def predict(w, x):
    """Given weight vector w, and feature set x, predict and return label y_hat."""
    y_hat = np.dot(w,x)
    return y_hat

if __name__ == '__main__':
    print 'Testing...a1.py'

    # load data
    red_data = np.genfromtxt(red_file, delimiter=';')[1:]           
    white_data = np.genfromtxt(white_file, delimiter=';')[1:]

    # preliminary analysis
    analyze(red_data)

    red_data_std = standardize(red_data)

    analyze(red_data_std)

    num_for_testing = 100

    red_data_std_train = red_data_std[num_for_testing:]
    red_data_std_test = red_data_std[:num_for_testing]

    examples_train, labels_train = extract(red_data_std_train)
    examples_test, labels_test = extract(red_data_std_test)

    # print 'X:\n', examples
    # print 'y:\n', labels


    start = 0
    end = 8
    logres = 100

    reg_terms = np.logspace(start, end, num=logres, base=10)
    rmse_errors = []
    mad_errors = []

    for reg in reg_terms:
        w = regress_ridge(examples_train, labels_train, reg)
        # print 'w:\n', w
        # print len(w)


        # w = regress_slow(examples, labels)
        # print 'w slow:\n', w
        # print len(w)



        print predict(w, red_data_std[-1][:-1])
        print red_data_std[-1][-1]

        Xex = [np.zeros(11) for x in range(10)]
        print 'RMSE Test:', rmse(w, Xex, np.ones(10)), rmse(w, Xex, np.zeros(10))
        print 'MAD Test:', mad(w, Xex, np.ones(10)), mad(w, Xex, np.zeros(10))

        print 'RMSE Control:', rmse(np.ones(len(w))*np.random.uniform(), examples_test, labels_test)
        print 'MAD Control:', mad(np.ones(len(w))*np.random.uniform(), examples_test, labels_test)

        rmse_error = rmse(w, examples_test, labels_test)
        mad_error = mad(w, examples_test, labels_test)

        print 'RMSE:', rmse_error
        print 'MAD:', mad_error

        rmse_errors.append(rmse_error)
        mad_errors.append(mad_error)


    # plot it
    plt.semilogx(reg_terms, rmse_errors)

    # plot setup
    plt.xlabel('Regularization Term (lambda)')
    plt.ylabel('RMSE')
    plt.title('RMSE as Regularization Term Varies')
    plt.grid(True)
    # plt.savefig("RMSEvsLambda.png")
    plt.show()



    # plot it
    plt.semilogx(reg_terms, mad_errors)

    # plot setup
    plt.xlabel('Regularization Term (lambda)')
    plt.ylabel('MAD')
    plt.title('MAD as Regularization Term Varies')
    plt.grid(True)
    # plt.savefig("MADvsLambda.png")
    plt.show()
    

