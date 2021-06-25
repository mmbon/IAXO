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


class IAXO_config:
    def __init__(self, length, angle, pressure, temperature): #Length in m, angle in degrees, pressure in Pa, temp in K
        self.length = length
        self.angle = angle
        self.height = calc_height(angle,length)
        self.pressure = pressure
        self.temperature = temperature

        self.set_density()
        self.set_areas_of_IAXO()

    def set_density(self):
        self.density = (self.pressure * M)/(con.R * self.temperature)

    def set_areas_of_IAXO(self):
        self.slices = []

        for i in range(int(self.length / 0.01)):
            self.slices.append(area_of_IAXO(self.length * 0.01, (i+0.5)*0.01))

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


class area_of_IAXO:
    def __init__(self,width,current_length):
        self.width = width
        self.current_length = current_length

    def set_density(self,density):
        self.density = density

    def set_next_area(self, next_object):
        self.next_object = next_object

    def setDensity(self,density):
        self.density = density

tester = IAXO_config(7.5,10,10000,300)

for i in range(len(tester.slices)):
    print('Meine Dichte ist: ', tester.slices[i].density)

