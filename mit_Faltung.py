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
import scipy.integrate as integrate
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

#Units
kappa = 1.67
M = 4.002602 #g/mol
temperature = 293.15 #in K
g_ay = 1e-11 * 1e-9 #1/eV

@njit
def pressureToDensity(pressure): #enter Pa
    return (pressure * M)/(con.R * temperature) * 1e-6  #returns g/cm^3

@njit
def attnuation_coeff(energy): #energy in keV
    return -1.5832+5.9195 * np.exp(-0.353808 * energy)+4.03598 * np.exp(-0.970557 * energy) #return cm^2/g

@njit
def height(x,angle,complete_Length):
    hp = np.sin(np.radians(angle))
    res = (complete_Length-x) * hp
    return res #returns height in m

@njit
def density(x,angle,complete_Length,pressure):
    hp = pressureToDensity(pressure) * (1-(kappa-1) / kappa * (M * con.g * height(x,angle,complete_Length)) / (con.R * temperature * 1e3))**(1/(kappa-1))
    # print(hp) # Hier geht ein Faktor von 1/10 ab
    return(hp)  #returns g/cm^3

@njit
def gamma_m(x,energy,angle,complete_Length,pressure):
    hp = np.exp(attnuation_coeff(energy))
    return hp * density(x,angle,complete_Length,pressure) * 100 #returns 1/m

@njit
def gamma_eV(x,energy,angle,complete_Length,pressure):
    hp = np.exp(attnuation_coeff(energy))
    return hp * density(x,angle,complete_Length,pressure) * 100.0 * 5067730.7 #returns eV

@njit
def P_axion_photon(x,energy,B,angle,complete_Length,pressure):
    B = B * 195.35277 #to convert to eV^2
    hp = ((g_ay*B)/2)**2 * (4/(gamma_eV(x,energy,angle,complete_Length,pressure)**2)) # der Vorfaktor mit den Gammas
    hp2 = 1 + np.exp(-gamma_m(x,energy,angle,complete_Length,pressure) * x) - np.exp(-gamma_m(x,energy,angle,complete_Length,pressure) * x / 2) #Die summe in den Klammern
    return hp * hp2

##The following is the derivative and therefore the probability distribution
@njit
def P_axion_photon_derivative(x,energy,B,angle,complete_Length,pressure):
    B = B * 195.35277  #to convert to eV^2
    g_eV = gamma_eV(x,energy,angle,complete_Length,pressure) #in eV
    g_m = gamma_m(x,energy,angle,complete_Length,pressure) # in 1/m
    hp = ((g_ay * B) / 2)**2 * (4 / (g_eV**2))  # der Vorfaktor mit gamma (einheitenlos)
    hp2 = 2/g_m + 2/g_m * np.exp(-g_m*x) - x * np.exp(-g_m*x) - 4/g_m * np.exp(-g_m*x/2) + x * np.exp(-g_m*x/2) #Einheit m
    hp3 = np.exp(attnuation_coeff(energy)) * 1e-4 # Einheit cm^2/g
    hp4 = pressureToDensity(pressure) * 1e6 # g/cm^3
    hp5 = 1/(kappa-1) * (1 - (kappa-1) / kappa * (M * con.g * height(x,angle,complete_Length)) / (con.R * temperature * 1e3))**(1/(kappa-1) - 1) * (kappa-1) / kappa * (M * con.g)/(con.R * temperature * 1e3) # Einheit: 1/m
    hp6 = np.sin(np.radians(angle)) #einheitenlos
    hp7 = hp3 * hp4 * hp5 * hp6 #1/m^2
    hp8 = hp7 * hp * hp2
    hp9 = ((g_ay * B) / 2)**2 * (4/g_eV)  # der Vorfaktor mit nur einem gamma, Einheit: 1/m
    hp10 = np.exp(-g_m*x) - np.exp(-g_m*x/2) #einheitenlos

    # print(1,hp)
    # print(2,hp2)
    # print(3,hp3)
    # print(4,hp4)
    # print(5,hp5)
    # print(6,hp6)
    # print(7,hp7)
    # print(8,hp8)
    # print(9,hp9)
    # print(10,hp10)
    # print(g_m)
    # print(pressureToDensity(pressure))

    return hp8 - (hp9 * hp10)


def numerical_derivative(x,precision,energy,B,angle,complete_Length,pressure):
    hp = P_axion_photon(x+precision,energy,B,angle,complete_Length,pressure)
    hp2 = P_axion_photon(x,energy,B,angle,complete_Length,pressure)
    return (hp-hp2)/precision #Warum negativ? Weil c = 0, automatisch


def integrate_numerical_derivative(start,stop,precision,energy,B,angle,complete_Length,pressure):
    hp = integrate.quad(numerical_derivative,start,stop,(precision,energy,B,angle,complete_Length,pressure))[0]
    #print(P_axion_photon_derivative(start,energy,B,angle,complete_Length,pressure))
    return hp + P_axion_photon(0,energy,B,angle,complete_Length,pressure) #Dieser Wert hat das Problem das P_axion_photon


def absorption_by_gas(x,energy,angle,complete_Length,pressure): #gives the probability of the photon over the length of x being absorbt by gas
    return np.exp(-gamma_m(x,energy,angle,complete_Length,pressure) * x)


def derivative_absorption_at_point_x(x,precision,energy,angle,complete_Length,pressure):
    hp = absorption_by_gas(x+precision,energy,angle,complete_Length,pressure)
    hp2 = P_axion_photon(x,energy,angle,complete_Length,pressure)
    return (hp-hp2) / precision


def integrate_numerical_absorption(start,stop,precision,energy,angle,complete_Length,pressure):
    hp = integrate.quad(derivative_absorption_at_point_x,start,stop,(precision,energy,angle,complete_Length,pressure))[0]
    return hp +



# print(P_axion_photon_derivative(4,5,1,0,20,100_000))
# print(numerical_derivative(4,1e-9,5,1,0,20,100_000))

# print(P_axion_photon(20,5,1,0,20,100_000))
# print(integrate.quad(P_axion_photon_derivative,0,20,(5,1,0,20,100_000)))
# print(integrate.quad(numerical_derivative,0,20,(1e-9,5,1,0,20,100_000)))

x_array = np.arange(0,25,0.1)
y_array = np.zeros(len(x_array))
y_array2 = np.zeros(len(x_array))

print(P_axion_photon(20,5,1,0,20,100_000))

for i in range(len(x_array)):
    y_array[i] = P_axion_photon(20,i/10,1,0,20,100_000)
    # y_array[i] = numerical_derivative(i/10,1e-9,5,1,0,20,100_000)
    y_array2[i] = integrate_numerical_derivative(0,20,1e-9,i/10,1,0,20,100_000)


fig1 = plt.figure(figsize=(12,8), dpi=80)
ax1 = fig1.add_axes([0.15,0.15,0.8,0.8])
#ax1.errorbar(x_disc_schwelle[1:],y_count_coincidence[1:],yerr=np.sqrt(y_count_coincidence[1:]), ls='none', capsize=2,elinewidth=0.5, capthick=0.5, color='k',label='Koinzidenz')
ax1.plot(x_array,y_array,color='red',label='Druck: 1 Bar')
ax1.plot(x_array,y_array2,color='green',label='Druck: 1 Bar')
# ax1.set_xlabel('Energie in keV')
# ax1.set_ylabel('Wahrscheinlichkeit einer Photonenmessung')
# ax1.legend(loc='upper left')
#plt.savefig('pics/Schwellenkurve.png')
plt.show()