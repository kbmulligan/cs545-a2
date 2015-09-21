# a2.py - 
# by K. Brett Mulligan
# CSU - CS545 (Fall 2015)
# Assignment 2


from __future__ import division

import math
import time
import numpy as np
from matplotlib import pyplot as plt

red_file = "winequality-red.csv"
white_file = "winequality-white.csv"

red_data = np.genfromtxt(red_file)
white_data = np.genfromtxt(white_file)

if __name__ == '__main__':
    print 'Testing...a1.py'

    