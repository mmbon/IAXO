#Leon Wietfeld

from multiprocessing import Pool
import os
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

energy_window = np.loadtxt("polypropylene_window_10micron.txt", skiprows=2,delimiter=',', usecols=0) * 0.001 #Energie in keV
transmission_coefficient_window = np.loadtxt("polypropylene_window_10micron.txt", skiprows=2,delimiter=',', usecols=1) #Transmission

energy_flux = np.loadtxt("axion_gae_flux.txt", skiprows=10,delimiter=',', usecols=0) #Energie in keV
flux_per_keV = np.loadtxt("axion_gae_flux.txt", skiprows=10,delimiter=',', usecols=1) #Flux 1/(10^19,keV*cm^2*day)

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
    return hp * density(x,angle,complete_Length,pressure) * 100.0 * 1.9732698e-07 #returns eV

@njit
def P_axion_photon(x,energy,B,angle,complete_Length,pressure):
    B = B * 195.35277 #to convert to eV^2
    hp = ((g_ay*B)/2)**2 * (4/(gamma_eV(x,energy,angle,complete_Length,pressure)**2)) # der Vorfaktor mit den Gammas
    hp2 = 1 + np.exp(-gamma_m(x,energy,angle,complete_Length,pressure) * x) - np.exp(-gamma_m(x,energy,angle,complete_Length,pressure) * x / 2) #Die summe in den Klammern
    return hp * hp2

##The following is the derivative and therefore the probability distribution
# @njit
# def P_axion_photon_derivative(x,energy,B,angle,complete_Length,pressure):
#     B = B * 195.35277  #to convert to eV^2
#     g_eV = gamma_eV(x,energy,angle,complete_Length,pressure) #in eV
#     g_m = gamma_m(x,energy,angle,complete_Length,pressure) # in 1/m
#     hp = ((g_ay * B) / 2)**2 * (4 / (g_eV**2))  # der Vorfaktor mit gamma (einheitenlos)
#     hp2 = 2/g_m + 2/g_m * np.exp(-g_m*x) - x * np.exp(-g_m*x) - 4/g_m * np.exp(-g_m*x/2) + x * np.exp(-g_m*x/2) #Einheit m
#     hp3 = np.exp(attnuation_coeff(energy)) * 1e-4 # Einheit cm^2/g
#     hp4 = pressureToDensity(pressure) * 1e6 # g/cm^3
#     hp5 = 1/(kappa-1) * (1 - (kappa-1) / kappa * (M * con.g * height(x,angle,complete_Length)) / (con.R * temperature * 1e3))**(1/(kappa-1) - 1) * (kappa-1) / kappa * (M * con.g)/(con.R * temperature * 1e3) # Einheit: 1/m
#     hp6 = np.sin(np.radians(angle)) #einheitenlos
#     hp7 = hp3 * hp4 * hp5 * hp6 #1/m^2
#     hp8 = hp7 * hp * hp2
#     hp9 = ((g_ay * B) / 2)**2 * (4/g_eV)  # der Vorfaktor mit nur einem gamma, Einheit: 1/m
#     hp10 = np.exp(-g_m*x) - np.exp(-g_m*x/2) #einheitenlos
#
#     # print(1,hp)
#     # print(2,hp2)
#     # print(3,hp3)
#     # print(4,hp4)
#     # print(5,hp5)
#     # print(6,hp6)
#     # print(7,hp7)
#     # print(8,hp8)
#     # print(9,hp9)
#     # print(10,hp10)
#     # print(g_m)
#     # print(pressureToDensity(pressure))
#
#     return hp8 - (hp9 * hp10)


@njit
def numerical_derivative(x,precision,energy,B,angle,complete_Length,pressure):
    hp = P_axion_photon(x+precision,energy,B,angle,complete_Length,pressure)
    hp2 = P_axion_photon(x,energy,B,angle,complete_Length,pressure)
    return (hp-hp2)/precision #Warum negativ? Weil c = 0, automatisch



def integrate_numerical_derivative(start,stop,precision,energy,B,angle,complete_Length,pressure):
    hp = integrate.quad(numerical_derivative,start,stop,(precision,energy,B,angle,complete_Length,pressure))[0]
    return hp + P_axion_photon(0,energy,B,angle,complete_Length,pressure)


@njit
def absorption_by_gas(x,energy,angle,complete_Length,pressure): #gives the probability of the photon over the length of x being absorbt by gas
    return np.exp(-gamma_m(x,energy,angle,complete_Length,pressure) * (complete_Length-x))


@njit
def derivative_absorption_at_point_x(x,precision,energy,angle,complete_Length,pressure):
    hp = absorption_by_gas(x+precision,energy,angle,complete_Length,pressure)
    hp2 = absorption_by_gas(x,energy,angle,complete_Length,pressure)
    return (hp-hp2)/precision



def integrate_numerical_absorption(start,stop,precision,energy,angle,complete_Length,pressure):
    hp = integrate.quad(derivative_absorption_at_point_x,start,stop,(precision,energy,angle,complete_Length,pressure))[0]
    #  print(absorption_by_gas(0,energy,angle,complete_Length,pressure))
    return hp + absorption_by_gas(0,energy,angle,complete_Length,pressure)



def combined_particle_probability(x,precision,energy,B,angle,complete_Length,pressure):
    hp = integrate_numerical_absorption(x,complete_Length,precision,energy,angle,complete_Length,pressure)
    hp2 = integrate_numerical_derivative(0,x,precision,energy,B,angle,complete_Length,pressure)
    return hp * hp2



def final_particle_probability(energy,B,angle,complete_Length,pressure):
    res = integrate.quad(combined_particle_probability,0,complete_Length,(1e-6,energy,B,angle,complete_Length,pressure))[0]
    return res

# print(absorption_by_gas(10,5,0,20,100_000))
# print(integrate_numerical_absorption(0,10,1e-9,5,0,20,100_000))
#
#
# # print(P_axion_photon_derivative(4,5,1,0,20,100_000))
# # print(numerical_derivative(4,1e-9,5,1,0,20,100_000))
#
# print(P_axion_photon(20,5,1,0,20,100_000))
# # print(integrate.quad(P_axion_photon_derivative,0,20,(5,1,0,20,100_000)))
# # print(integrate.quad(numerical_derivative,0,20,(1e-9,5,1,0,20,100_000)))
#

def pp_fenster(energy): #takes enery in eV
    return np.interp(energy,energy_window,transmission_coefficient_window)

#to save computational time, we don't interpolate each time
f_ground = sc.interpolate.interp1d(energy_flux,flux_per_keV,fill_value="extrapolate",kind='quadratic')
def flux_interpolation(energy): #takes energy in keV
    return f_ground(energy)


# x_interpolated = np.linspace(min(energy_flux),max(energy_flux),len(energy_flux))
# f_ground = sc.interpolate.interp1d(energy_flux,flux_per_keV,kind='quadratic')
# y_interpolated = f_ground(x_interpolated)

def final_number_of_photons(x,B,angle,complete_Length,pressure): #takes Energy in keV, energy = x
    incoming = flux_interpolation(x) * 32.5**2 * np.pi * 365.25  # Using cm^2 and 365.25 to cancel out the units in the txt document
    converting = final_particle_probability(x,B,angle,complete_Length,pressure)
    window = pp_fenster(x*1e3) #needed to convert into eV
    return incoming * converting * window


x_array = np.linspace(0.0,10,10_000)

def calculate_distribution(B,angle,pressure):
    y_array = np.linspace(0.0,10.0,10_000)
    for i in range(10_000):
        y_array[i] = final_number_of_photons(y_array[i],B,angle,20,pressure)
    return y_array


def create_Data_array(variable,init_data):
    if variable == "Angle" or variable == "angle":
        vars = [0]*len(init_data)

        for i in range(len(init_data)):
            vars[i] = (1,init_data[i],100_000)

        with Pool(5) as p:
            hp = p.starmap(calculate_distribution,vars)

    elif variable == "B" or variable == "b":
        vars = [0] * len(init_data)

        for i in range(len(init_data)):
            vars[i] = (init_data[i],0,100_000)

        with Pool(5) as p:
            hp = p.starmap(calculate_distribution,vars)

    elif variable == "Pressure" or variable == "pressure":
        vars = [0] * len(init_data)

        for i in range(len(init_data)):
            vars[i] = (1,0,init_data[i])

        with Pool(5) as p:
            hp = p.starmap(calculate_distribution,vars)

    return hp


# total_expected_photons = integrate.quad(final_number_of_photons,0,10.0,(1,0,20,100_000))
# print(total_expected_photons)

plottable_list = create_Data_array("Angle",[0,15,35])

fig1 = plt.figure(figsize=(12,8), dpi=80)
ax1 = fig1.add_axes([0.15,0.15,0.8,0.8])
colour_list = ['r','k','g','b','m','y','c']
label_list = ['0 Grad','15 Grad','35 Grad']
# label_list = ['1 Tesla','2 Tesla','3 Tesla']
# label_list = ['50 kPa','100 kPa','200 kPa']
#ax1.errorbar(x_disc_schwelle[1:],y_count_coincidence[1:],yerr=np.sqrt(y_count_coincidence[1:]), ls='none', capsize=2,elinewidth=0.5, capthick=0.5, color='k',label='Koinzidenz')
for i in range(len(plottable_list)):
    ax1.plot(x_array,plottable_list[i],color=colour_list[i],label=label_list[i])

ax1.set_xlabel('Energie in keV')
ax1.set_ylabel('Detektierte Photonen')
ax1.legend(loc='upper right')
# plt.savefig('pics/Schwellenkurve.png')
plt.show()