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

red_data = np.genfromtxt(red_file, delimiter=';')[1:]           # remove the first one, because it's a label
white_data = np.genfromtxt(white_file, delimiter=';')[1:]


def analyze(data):
    print data
    print 'Length:', len(data)
    print 'Features:', len(data[0])

    scores = [x[11] for x in data]

    print 'Max:', max(scores)
    print 'Min:', min(scores)

    plt.hist(scores)
    plt.show()

if __name__ == '__main__':
    print 'Testing...a1.py'

    # analyze(red_data)
    # analyze(white_data)