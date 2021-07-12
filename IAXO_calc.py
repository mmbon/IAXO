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
M = 4.002602 #g/mol

length = 5 # in meters
precision = 10
step_length_diff = length/precision
angle = 0 #angle in degree
pressure_at_bottom = 100_000 #in Pa
temperature = 293 #in K
energy = 1e-3 #in keV
gay = 1000000 # Einheiten?
B = 100000 # Einheiten?

height = np.zeros(precision)
density = np.zeros(precision)
gammas = np.zeros(precision)
probability_creation = np.zeros(precision)

@njit
def barometric_formula(height_diff,density_h0): #enter g/cm^3
    return density_h0 * (1 - (kappa-1)/kappa * (M * con.g * height_diff)/(con.R*temperature*1e3))**((kappa-1)/kappa) #returns g/cm^3

@njit
def pressureToDensity(pressure): #enter Pa
    return (pressure * M * 1e6) / (con.R * temperature) #returns g/cm^3

@njit
def attnuation_coeff(): #energy in keV
    return -1.5832+5.9195 * np.exp(-0.353808 * energy)+4.03598 * np.exp(-0.970557 * energy) #return g/cm^2

@njit
def conversion_probability(gamma,L):
    print(L)
    return ((gay*B)/2)**2 * 1/(gamma**2/4) * (1+np.exp(- gamma * L)-2*np.exp(-(gamma * L)/2))

def calcHeigth(): #calculates in m
    hp = np.sin(np.radians(angle))
    for i in range(len(height)):
        height[i] = hp*(length - (step_length_diff*(i+1)))

def calcDensity():
    density[len(density)-1] = pressureToDensity(pressure_at_bottom)
    height_diff = step_length_diff * np.sin(np.radians(angle))

    for i in reversed(range(len(density)-1)):
        density[i] = barometric_formula(height_diff,density[i+1])

def calcGamma():
    koeff = np.exp(attnuation_coeff())
    for i in range(len(gammas)):
        gammas[i] = density[i]*koeff * 1e2 #gives gamma in 1/m

def calcConvProbAxion():
    probability_creation[0] = conversion_probability(gammas[0],step_length_diff)
    for i in range(1,len(probability_creation)):
        probability_creation[i] += conversion_probability(gammas[i],(i+1)*step_length_diff) - conversion_probability(gammas[i-1],i * step_length_diff)


calcHeigth()
calcDensity()
calcGamma()
gammas = gammas/1e13
calcConvProbAxion()

print(probability_creation)
print(sum(probability_creation))
print(conversion_probability(gammas[len(gammas)-1],length))

