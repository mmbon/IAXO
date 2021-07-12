#Leon Wietfeld
import math
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#import numpy.string
import numpy as np
import sympy as sy
import scipy.constants as con
from lmfit import Model as md
#import pyprac
import scipy as sc
import scipy.optimize as optimize
import scipy.stats as stat
from uncertainties import ufloat
from uncertainties.umath import *
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def testfunc1(x):
    return x**2

def conversion_probability(gamma,L,gay,B):
    return ((gay*B)/2)**2 * 1/(gamma**2/4) * (1+np.exp(- gamma * L)-2*np.exp(-(gamma * L)/2) )

def wrapper_for_conversion(x):
    return conversion_probability(7.06424619e+13,x,1,1)

def test_slices(length, number_slices):
    step = length/number_slices
    probability = wrapper_for_conversion(step)

    for i in range(1,number_slices):
        probability += wrapper_for_conversion(step*(i+1))-wrapper_for_conversion(step*i)
        # print(wrapper_for_conversion(step*(i+1))-wrapper_for_conversion(step*i))

    return probability

print(wrapper_for_conversion(7.5))
print(test_slices(7.5,1000))


