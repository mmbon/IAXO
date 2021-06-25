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

kappa = 1.67
M = 4


def calc_height(angle, length): #angle in degree
    return np.sin(math.radians(angle)) * length


def barometric_formula(height_diff, temp,density_h0):
    return density_h0 * (1 - (kappa-1)/kappa * (M * con.g * height_diff)/(con.R*temp))**((kappa-1)/kappa)


def reverse_barometric_formula(height_diff, temp,density_h1):
    return density_h1 * (1 - (kappa-1)/kappa * (M * con.g * height_diff)/(con.R*temp))**(-(kappa-1)/kappa)

def attnuation_coeff(energy):
    return -1.5832+5.9195 * np.exp(-0.353808 * energy)+4.03598 * np.exp(-0.970557 * energy)

def conversion_probability(gamma,L,gay,B):
    return ((gay*B)/2)**2 * 1/(gamma**2/4) * (1+np.exp(- gamma * L)-2*np.exp(-(gamma * L)/2) )

class IAXO_config:
    def __init__(self, length, angle, pressure, temperature,energy,g_ay,B): #Length in m, angle in degrees, pressure in Pa, temp in K,energy in keV
        self.length = length
        self.angle = angle
        self.height = calc_height(angle,length)
        self.pressure = pressure
        self.temperature = temperature
        self.energy = energy
        self.g_ay = g_ay
        self.B = B

        self.set_density()
        self.set_areas_of_IAXO()

    def set_density(self):
        self.density = (self.pressure * M)/(con.R * self.temperature)

    def set_areas_of_IAXO(self):
        self.slices = []

        for i in range(int(self.length / 0.01)):
            self.slices.append(area_of_IAXO(self.length * 0.01, (i+0.5)*0.01, self.B,self.g_ay))

        for i in range(len(self.slices)-1):
            self.slices[i].set_next_area(self.slices[i+1])

        self.slices[len(self.slices)-1].set_next_area(None)

        self.set_slice_density()

    def set_slice_density(self):
        height_diff = calc_height(self.angle,self.length)/2

        rho_0 = reverse_barometric_formula(height_diff,self.temperature,self.density)
        rho_0 = barometric_formula(calc_height(self.angle,self.slices[len(self.slices)-1].width/2),self.temperature,rho_0)
        rho_1 = 0

        for i in range(len(self.slices)):
            self.slices[(len(self.slices))-(i+1)].setDensity(rho_0)
            rho_1 = barometric_formula(calc_height(self.angle,self.slices[i].width),self.temperature,rho_0)
            rho_0 = rho_1

        self.set_slice_attenuation_coef()

    def set_slice_attenuation_coef(self):
        for i in range(len(self.slices)):
            self.slices[i].set_attenuation_coeff(self.energy)

    def calculate_axion_conversion(self,starting_axions):
        self.slices[0].calc_converted_photons(starting_axions)

class area_of_IAXO:
    def __init__(self,width,current_length,B,g_ay):
        self.width = width
        self.current_length = current_length
        self.B = B
        self.g_ay = g_ay

    def set_density(self,density):
        self.density = density

    def set_next_area(self, next_object):
        self.next_object = next_object

    def setDensity(self,density):
        self.density = density

    def set_attenuation_coeff(self,energy):
        hp = np.exp(attnuation_coeff(energy))*1/10
        self.attenuation_coefficient = hp * self.density

    def calc_converted_photons(self,number_axions):
        self.created_photons = number_axions * conversion_probability(self.attenuation_coefficient,self.width,self.g_ay,self.B)
        self.remaining_axions = number_axions - self.created_photons

        if self.next_object is not None:
            return self.next_object.calc_converted_photons(self.remaining_axions)

tester = IAXO_config(7.5,10,10000,300,511,1,1)
tester.calculate_axion_conversion(100000)

for i in range(len(tester.slices)):
    print('Anzahl der erzeugten Photonen ist: ', tester.slices[i].created_photons)
    print('Anzahl der übrigbleibenden Axionen ist: ',tester.slices[i].remaining_axions)

