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
import timeit
from numba import njit
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# starttime = timeit.default_timer()
# calcHeigth()
# print("The time difference is :", timeit.default_timer() - starttime)

kappa = 1.67
M = 4.002602

length = 7.5 # in meters
precision = 10_000
number_of_slices = length/precision
angle = 0 #angle in degree
pressure_at_bottom = 100_000 #in Pa

height = np.empty(round(number_of_slices+1))

@njit
def calcHeigth():
    hp = np.sin(np.radians(angle))
    for i in range(len(height)):
        height[i] = precision*i*hp

@njit
def barometric_formula(height_diff, temp,density_h0):
    return density_h0 * (1 - (kappa-1)/kappa * (M * con.g * height_diff)/(con.R*temp))**((kappa-1)/kappa)