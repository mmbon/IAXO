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
import timeit
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def Func(x):
    return 1/10 * x**2

def func(x_0):
    prec = 1e-12
    hp = Func(x_0+prec) - Func(x_0)
    return hp/prec


print(Func(3))

final = 0
prec = 0.1
for i in np.arange(0,3,prec):
    print(i,i+prec/2)
    print(func(i+prec/2))
    print('----------------------')
    final += (1-Func(i))*func(i+prec/2)

print(final)

# starttime = timeit.default_timer()
# print(func(1))
# print("The time difference is :", timeit.default_timer() - starttime)

# def testfunc1(x):
#     return x**2
#
# def conversion_probability(gamma,L,gay,B):
#     return ((gay*B)/2)**2 * 1/(gamma**2/4) * (1+np.exp(- gamma * L)-2*np.exp(-(gamma * L)/2) )
#
# def wrapper_for_conversion(x):
#     return conversion_probability(7.06424619e+13,x,1,1)
#
# def test_slices(length, number_slices):
#     step = length/number_slices
#     probability = wrapper_for_conversion(step)
#
#     for i in range(1,number_slices):
#         probability += wrapper_for_conversion(step*(i+1))-wrapper_for_conversion(step*i)
#         # print(wrapper_for_conversion(step*(i+1))-wrapper_for_conversion(step*i))
#
#     return probability
#
# print(wrapper_for_conversion(7.5))
# print(test_slices(7.5,1000))


