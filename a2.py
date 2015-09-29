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

    def __init__(self, filename, name=''):
        self.winedata = np.genfromtxt(filename, delimiter=';')[1:]       # remove the first line (featureset description)
        if (name == ''):
            self.name = filename
        else:
            self.name = name
        self.w = []

    def __len__(self):
        return len(self.winedata)

    def split(self, percent_test):
        """Splits data into training and testing based on percent_test."""
        num_for_testing = int(np.floor(len(self) * percent_test))
        self.data_train = self.winedata[num_for_testing:]
        self.data_test = self.winedata[:num_for_testing]

    def standardize(self, data=[]):
        """Given matrix data, standardizes each feature by subtracting mean and
        dividing by standard deviation. Returns new matrix of same shape with 
        standardized values."""

        if (data == []):
            data = self.winedata

        data = np.insert(data, 0, 1, axis=1)                # insert ones for bias

        self.winedata_raw = np.copy(data)
        new_data = np.copy(np.transpose(data))


        self.means = np.zeros(len(new_data))
        self.std_devs = np.zeros(len(new_data))

        for i in range(len(new_data)):
            self.means[i] = np.mean(new_data[i])
            self.std_devs[i] = np.std(new_data[i])
            if (self.std_devs[i] != 0):                                                         # make sure not to divide by zero
                new_data[i] = [(x - self.means[i])/self.std_devs[i] for x in new_data[i]]
            else:
                # new_data[i] = [(x - self.means[i]) for x in new_data[i]]
                pass
        
        self.winedata = np.transpose(new_data)
        print self.winedata
        return self.winedata

    def unstandardize_value(self, val, index):
        """Given feature number 'index' and standardized value 'val', return original value
        by looking up mean and std dev and adding and multiplying them back respectively."""
        return val * self.std_devs[index] + self.means[index]

    def predict_null(self, w, x):
        """Given weight vector w, and feature set x, predict and return CONSTANT label y_hat. y_hat 
        is equal to mean label of training data."""
        y_hat = np.mean(self.data_train.T[-1])
        return y_hat

    def predict(self, w, x):
        """Given weight vector w, and feature set x, predict and return label y_hat."""
        y_hat = np.dot(w,x)
        return y_hat

    def mad(self, w, data, labels):
        """Mean Absolute Deviation. Iterate over all examples in 'data' and find the absolute difference label and
        predicted label. Sum all this error, then find the arithmetic mean by dividing by 
        total number of data examples. Return this value."""

        total = 0
        N = len(data)
        for i in range(N):
            error = labels[i] - self.predict(w, data[i])
            total += np.absolute(error)

        return total/float(N)

    def rmse(self, w, data, labels):
        """Root Mean Square Error. Iterate over all examples in 'data' and find the difference between label and 
        predicted label. Square this error and total it for all data, then find the arithmetic
        average and take the square root. Return that value."""

        total = 0
        N = len(data)
        for i in range(N):
            error = labels[i] - self.predict(w, data[i])
            total += np.square(error)

        return np.sqrt(total/float(N))

    ### END CLASS #####################################################################################################

### UTILITY ###########################################################################################################

def analyze(data, plot_this=False):
    print data
    print 'Data points        :', len(data)
    print 'Features per point :', len(data[0])

    scores = [x[-1] for x in data]

    print 'Max:', max(scores)
    print 'Min:', min(scores)

    if (plot_this == True):
        plt.hist(scores)
        plt.show()

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


# REC CURVES ##########################################################################################################

def rec_curve(dataset, weight_vectors, Xtest, ytest):
    """Plot REC Curve given predicted labels, true labels, and max tolerance."""

    tolerance_step = 0.01
    tolerance_start = 0
    tolerance_stop = 3

    tolerances = np.arange(tolerance_start,tolerance_stop,tolerance_step)
    accuracies = np.zeros(len(tolerances))

    total_test_points = len(Xtest)

    
    for w in weight_vectors:
        predictions = [dataset.predict(w, Xtest[i]) for i in range(total_test_points)]

        # iterate through tolerances and calculate accuracy for each
        for i in range(len(tolerances)):
            accuracies[i] = accuracy(ytest, predictions, tolerances[i])

        # plot it
        plt.plot(tolerances, accuracies)


    # always include null model
    predictions = [dataset.predict_null(weight_vectors[0], Xtest[i]) for i in range(total_test_points)]

    # iterate through tolerances and calculate accuracy for each
    for i in range(len(tolerances)):
        accuracies[i] = float(accuracy(ytest, predictions, tolerances[i]))

    # plot it
    plt.plot(tolerances, accuracies)


    # plot setup
    plt.xlabel('Tolerance ($\epsilon$)')
    plt.ylabel('Accuracy')
    plt.title('REC' + ' (' + dataset.name + ')')
    plt.grid(True)
    plt.savefig('REC' + '(' + dataset.name + ').png')
    plt.show()
    plt.close()

    return

def accuracy(labels_true, labels_predicted, tolerance, square_error=False):
    """Sum the number of predicted labels within 'tolerance' of true labels and divide by total. Return this fraction."""
    correct_results = np.zeros(len(labels_true))
    for i in range(len(labels_true)):
        if (square_error):
            diff = np.square(labels_predicted[i] - labels_true[i])
        else:
            diff = np.absolute(labels_predicted[i] - labels_true[i])
        # print labels_true[i]
        # print labels_predicted[i]
        # print 'diff=',diff

        if (diff <= tolerance):
            correct_results[i] = 1
        else:
            correct_results[i] = 0

    correct = np.sum(correct_results)
    total = float(len(labels_true))
    acc = correct/total

    # print 'Accuracy'
    # print 'correct =', correct
    # print 'total =', total
    # print 'acc =', acc
    # print type(acc)
    return acc


# PART 2: Pearson Product-Moment Correlation Coefficient ##############################################################

def ppmcc(x, y):
    """Calculate and return Pearson product-moment correlation coefficient, r.
    Given standardized vectors x and y, sum the products, and divide by n - 1."""
    r = 0

    if (len(x) != len(y)):
        print '\n\nPPMCC Error: x and y different length.\n\n'
    else:
        n = len(x)
        products = [x[i] * y[i] for i in range(n)]
        r = np.sum(products)/float(n - 1)

    return r


def regress_and_find_errors (dataset, Xtrain, ytrain, Xtest, ytest):

    start = -3
    end = 8
    logres = 100

    reg_terms = np.logspace(start, end, num=logres, base=10)
    rmse_errors = []
    mad_errors = []

    best_reg_value_rmse = 0
    best_reg_value_mad = 0

    for reg in reg_terms:
        w = regress_ridge(Xtrain, ytrain, reg)
        # print 'w:\n', w
        # print len(w)

        # w = regress_slow(examples, labels)
        # print 'w slow:\n', w
        # print len(w)

        # print predict(w, red_data_std[-1][:-1])
        # print red_data_std[-1][-1]

        # Xex = [np.zeros(len(Xtrain)) for x in range(10)]
        # print 'RMSE Test:', rmse(w, Xex, np.ones(10)), rmse(w, Xex, np.zeros(10))
        # print 'MAD Test:', mad(w, Xex, np.ones(10)), mad(w, Xex, np.zeros(10))

        # print 'RMSE Control:', rmse(np.ones(len(w))*np.random.uniform(), Xtest, ytest)
        # print 'MAD Control:', mad(np.ones(len(w))*np.random.uniform(), Xtest, ytest)

        rmse_error = dataset.rmse(w, Xtest, ytest)
        mad_error = dataset.mad(w, Xtest, ytest)

        # print 'RMSE:', rmse_error
        # print 'MAD:', mad_error

        rmse_errors.append(rmse_error)
        mad_errors.append(mad_error)

        if rmse_error <= min(rmse_errors):
            best_reg_value_rmse = reg

        if mad_error <= min(mad_errors):
            best_reg_value_mad = reg


    best_reg_value_avg = (best_reg_value_rmse + best_reg_value_mad) / 2.0       # find avg between RMSE and MAD

    print 'Optimal reg term(RMSE):', best_reg_value_rmse
    print 'Optimal reg term(MAD):', best_reg_value_mad
    print 'Average:', best_reg_value_avg



    # PLOT BEST REGRESSION ############################################################################################
    w = regress_ridge(Xtrain, ytrain, best_reg_value_mad)

    show_this_many = 40

    Xdisplay = Xtest[:show_this_many]
    ydisplay = ytest[:show_this_many]

    predictions = np.array([dataset.predict(w, Xdisplay[i]) for i in range(len(Xdisplay))])

    X = np.arange(len(Xdisplay))
    predictions = (predictions * dataset.std_devs[-1]) + dataset.means[-1]
    ydisplay = (ydisplay * dataset.std_devs[-1]) + dataset.means[-1]
    difference = np.absolute(predictions - ydisplay)
    
    # plot true and predicted labels, true emphasized with scatter dots
    plt.plot(X, ydisplay, X, predictions)
    plt.plot(X,difference)
    plt.scatter(X, ydisplay)
    plt.axis([0,len(X),0,10])


    # plot setup
    plt.xlabel('Test Data Point')
    plt.ylabel('Score')
    plt.title('Sample Predicted and True Scores' + ' (' + dataset.name + ')')
    plt.grid(True)
    plt.savefig('Predicted and True Scores' + '(' + dataset.name + ').png')
    plt.show()
    plt.close()



    ### PLOT ERROR VS REGULARIZATION ###################################################################################

    # plot it
    plt.semilogx(reg_terms, [error for error in rmse_errors])                           # normalized
    # plt.semilogx(reg_terms, [dataset.std_devs[-1]*error for error in rmse_errors])    # unnormalized

    # plot setup
    plt.xlabel('Regularization Term ($\lambda$)')
    plt.ylabel('RMSE')
    plt.title('RMSE as Regularization Term Varies' + ' (' + dataset.name + ')')
    plt.grid(True)
    plt.savefig('RMSEvsLambda' + '(' + dataset.name + ').png')
    plt.show()
    plt.close()



    # plot it
    plt.semilogx(reg_terms, [error for error in mad_errors])                            # normalized
    # plt.semilogx(reg_terms, [dataset.std_devs[-1]*error for error in mad_errors])     # unnormalized

    # plot setup
    plt.xlabel('Regularization Term ($\lambda$)')
    plt.ylabel('MAD')
    plt.title('MAD as Regularization Term Varies' + ' (' + dataset.name + ')')
    plt.grid(True)
    plt.savefig('MADvsLambda' + '(' + dataset.name + ').png')
    plt.show()
    plt.close()


    return best_reg_value_mad


def evaluate_feature_importance(dataset):
    reg = 1

    training_subset_fraction = 0.5
    total = len(dataset.data_train)
    cutoff = int(np.floor(total * training_subset_fraction))

    subset_train = dataset.data_train[:cutoff]

    Xsubset, ysubset = extract(subset_train)
    # print subset_train, len(subset_train)

    w = regress_ridge(Xsubset, ysubset, reg)

    print w

    # index_by_importance = list(reversed(sorted(range(len(w)), key=lambda k: w[k])))
    index_and_weight = [(i, w[i]) for i in range(len(w))]

    print index_and_weight

    X = np.arange(len(w))

    coefs = [ppmcc(Xsubset.T[i], ysubset) for i in X]

    print 'Correlation coefficients:', coefs, len(coefs)

    plt.scatter(w, coefs)


    # plot setup
    plt.xlabel('Weight Vector Value')
    plt.ylabel('Correlation Coefficient ($r$)')
    plt.title('Weight Vector vs Correlation Coefficients' + ' (' + dataset.name + ')')
    plt.grid(True)
    plt.savefig('WeightVectorCorrelation' + '(' + dataset.name + ').png')
    plt.show()
    plt.close()





    # incrementally remove features, retrain, and save RMSE and MAD





    # plot RMSE and MAD against feature removal





    return



def test_dataset(dataset):
    """Run full test on dataset. Test includes regression for range of lambda, REC curves, and 
    analysis of feature importance."""

    ### SETUP #########################################################################################################

    reserve = 0.33

    # initial and only shuffle - makes each call to test_dataset unique
    # np.random.shuffle(dataset.winedata)

    # analyze before and after standardization
    analyze(dataset.winedata)
    dataset.standardize()
    analyze(dataset.winedata)

    dataset.split(reserve)

    examples_train, labels_train = extract(dataset.data_train)
    examples_test, labels_test = extract(dataset.data_test)


    ### PLOT RMSE AND MAD AS FUNCTION OF REGULARIZATION TERM ##########################################################
    # best_reg = regress_and_find_errors(dataset, examples_train, labels_train, examples_test, labels_test)


    ### PLOT REC: ACCURACY VS TOLERANCE ###############################################################################
    one_reg = 1
    mid_reg = 5000
    high_reg = 20000
    low_reg = 0.0001

    weights = []
    weights.append(regress_ridge(examples_train, labels_train, one_reg))
    # w.append(regress_ridge(examples_train, labels_train, high_reg))
    # w.append(regress_ridge(examples_train, labels_train, low_reg))
    weights.append(regress_ridge(examples_train, labels_train, mid_reg))

    # rec_curve(dataset, weights, examples_test, labels_test)


    ### EVALUATING FEATURE IMPORTANCE #################################################################################
    evaluate_feature_importance(dataset)

    
    return



if __name__ == '__main__':
    print 'Testing...a1.py'

    # load data
    red = WineDataset(red_file, 'red')
    white = WineDataset(white_file, 'white')

    test_dataset(red)
    test_dataset(white)
