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
angle = 20 #angle in degree
pressure_at_bottom = 100_000 #in Pa
temperature = 293 #in K
energy = 1 #in keV
gay = 1 # 1/eV
B = 1 * 195.35277 # eV^2

height = np.zeros(precision)
density = np.zeros(precision)
gammas = np.zeros(precision)
probability_creation = np.zeros(precision)
final_photons = np.zeros(precision)

@njit
def barometric_formula(height_diff,density_h0): #enter g/cm^3
    return density_h0 * (1 - (kappa-1)/kappa * (M * con.g * height_diff)/(con.R*temperature*1e3))**((kappa-1)/kappa) #returns g/cm^3

@njit
def pressureToDensity(pressure): #enter Pa
    return (pressure * M * 1e6) / (con.R * temperature) #returns g/cm^3

@njit
def attnuation_coeff(energy): #energy in keV
    return -1.5832+5.9195 * np.exp(-0.353808 * energy)+4.03598 * np.exp(-0.970557 * energy) #return g/cm^2

@njit
def conversion_probability(gamma,L,B):
    return ((gay*B)/2)**2 * 1/(gamma**2/4) * (1+np.exp(- gamma * L)-2*np.exp(-(gamma * L)/2))

def calcHeigth(angle): #calculates in m
    hp = np.sin(np.radians(angle))
    for i in range(len(height)):
        height[i] = hp*(length - (step_length_diff*(i+1)))


def calcDensity(angle):
    density[len(density)-1] = pressureToDensity(pressure_at_bottom)
    height_diff = step_length_diff * np.sin(np.radians(angle))

    for i in reversed(range(len(density)-1)):
        density[i] = barometric_formula(height_diff,density[i+1])


def calcGamma(eneg):
    koeff = np.exp(attnuation_coeff(eneg))
    for i in range(len(gammas)):
        gammas[i] = density[i]*koeff * 1e2 * 1.97e-7 #gives gamma in 1/eV


def calcConvProbAxion(B):
    probability_creation[0] = conversion_probability(gammas[0],step_length_diff / 5067730.7,B) #Crude way to implement length in eV
    for i in range(1,len(probability_creation)):
        probability_creation[i] = conversion_probability(gammas[i],(i+1)*step_length_diff / 5067730.7,B) - conversion_probability(gammas[i-1],i * step_length_diff / 5067730.7,B)


def calcAbsorption():
    for i in range(len(probability_creation)):
        final_photons[i] = probability_creation[i]
        for j in range(i+1,len(probability_creation)):
            final_photons[i] = final_photons[i] * np.exp(- gammas[j] * step_length_diff / 5067730.7)

# calcHeigth()
# calcDensity()
# calcGamma()
# calcConvProbAxion()
# calcAbsorption()

save_array = np.eye(5, len(np.arange(0.001,5,0.001)))
print(save_array)
x = np.arange(0.001,5,0.001)

for j in range(5):
    for i in range(len(x)):
        calcHeigth(0)
        calcDensity(0)
        calcGamma(x[i])
        calcConvProbAxion(j * 195.35277)
        calcAbsorption()

        save_array[j][i] = np.sum(final_photons)


fig1 = plt.figure(figsize=(12,8), dpi=80)
ax1 = fig1.add_axes([0.15,0.15,0.8,0.8])
#ax1.errorbar(x_disc_schwelle[1:],y_count_coincidence[1:],yerr=np.sqrt(y_count_coincidence[1:]), ls='none', capsize=2,elinewidth=0.5, capthick=0.5, color='k',label='Koinzidenz')
ax1.plot(x,save_array[0],color='red',label='Magnetfeld: 0 T')
ax1.plot(x,save_array[1],color='cyan',label='Magnetfeld: 1 T')
ax1.plot(x,save_array[2],color='blue',label='Magnetfeld: 2 T')
ax1.plot(x,save_array[3],color='lime',label='Magnetfeld: 3 T')
ax1.plot(x,save_array[4],color='black',label='Magnetfeld: 4 T')
ax1.set_xlabel('Energie in keV')
ax1.set_ylabel('Wahrscheinlichkeit einer Photonenmessung')
ax1.legend(loc='upper left')
#plt.savefig('pics/Schwellenkurve.png')
plt.show()