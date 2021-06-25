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

temperature = 293.15
kappa = 1.67
M = 4
g = 9.81
R = 8.314

def calc_height(angle, length): #angle in degree
    return np.sin(math.radians(angle)) * length


def attenuation_photons(energy): #Energy in keV
    return -1.5832 + 5.9195 * np.exp(-0.353808 * energy) + 4.03598 * np.exp(-0.970557 * energy)


def intensity_decay(I_0,density,distance,attenuation): #100 is used to convert from cm to m
    return I_0 * np.exp(- 100 * attenuation * density * distance)


def barometric_formula(height_diff, temp,density_h0):
    return density_h0 * (1- (kappa-1)/kappa * (M * g * height_diff)/(R*temp))**((kappa-1)/kappa)


def reverse_barometric_formula(height_diff, temp,density_h1):
    return density_h1 * (1- (kappa-1)/kappa * (M * g * height_diff)/(R*temp))**(-(kappa-1)/kappa)


def density(pressure): #Input in Pa, output in g/cm^3
    pressure = pressure
    return (pressure / 1000) * (M / (R * temperature * 1000))


class IAXO_config:
    def __init__(self, length, angle): #Length in m, angle in degrees
        self.length = length
        self.angle = angle
        self.height = calc_height(angle,length)

    def configuration(self,setup,setuplengths,setupPressure): #Setup = names of materials, setuplengths = lengths of the materials in m, pressure in Pa
        self.obj_list = []
        for i in range(len(setup)):

            self.obj_list.append(homogenious_Material(setup[i],setuplengths[i],setupPressure[i],self.angle))


        if sum(setuplengths) != self.length:
            raise print('The length of the modules does not fit the whole length')

        self.link_objects()

    def link_objects(self):
        for i in range(len(self.obj_list)-1):
            self.obj_list[i].add_following(self.obj_list[i+1])

        self.obj_list[len(self.obj_list)-1].add_following(None)

    def calc_transmission(self, photon_energy):
        self.photon_energy = photon_energy

        return self.obj_list[0].calc_transmission_rate(self.photon_energy,100)


class homogenious_Material:
    def __init__(self,material,length,pressure,angle):
        self.material = material
        self.length = length
        self.angle = angle
        self.pressure = pressure

        self.config_riemann_slices()

    def add_following(self, next_material):
        self.next_material = next_material

    def config_riemann_slices(self):
        self.slices = []

        for i in range(int(self.length/0.01)):
            self.slices.append(incremental_material(self.length*0.01))

        for i in range(len(self.slices)-1):
            self.slices[i].add_following(self.slices[i+1])

        self.slices[len(self.slices)-1].add_following(None)

        self.set_pressure()

    def set_pressure(self):
        height_diff = calc_height(self.angle,self.length)/2

        rho_0 = reverse_barometric_formula(height_diff,temperature,density(self.pressure))
        rho_1 = 0

        for i in range(len(self.slices)):
            self.slices[(len(self.slices)-1)-i].setPressure(rho_0)
            rho_1 = barometric_formula(calc_height(self.angle,self.slices[i].distance),temperature,rho_0)
            rho_0 = rho_1

    def calc_transmission_rate(self,energy, I_0_in):

        I_0_out = self.slices[0].calc_transmission(energy,I_0_in)

        if self.next_material is not None:
            return self.next_material.calc_transmission_rate(energy,I_0_out)
        else:
            return I_0_out


class incremental_material:
    def __init__(self,distance):
        self.distance = distance

    def setPressure(self,pressure):
        self.pressure = pressure

    def add_following(self,next_Slice):
        self.next_Slice = next_Slice

    def calc_transmission(self,energy,I_0_in):

        I_0_out = intensity_decay(I_0_in,self.pressure,self.distance,np.exp(attenuation_photons(energy)))

        if self.next_Slice is not None:
            return self.next_Slice.calc_transmission(energy,I_0_out)
        else:
            return I_0_out


sys.setrecursionlimit(50000)
tester = IAXO_config(5,0)
tester.configuration(['hell yeah'],[5],[1e5])
print(tester.calc_transmission(7))

x = np.arange(0.1,1000,1)
y = tester.calc_transmission(x)

fig1 = plt.figure(figsize=(12,8), dpi=80)
ax1 = fig1.add_axes([0.15,0.15,0.8,0.8])
#ax1.errorbar(x_disc_schwelle[1:],y_count_coincidence[1:],yerr=np.sqrt(y_count_coincidence[1:]), ls='none', capsize=2,elinewidth=0.5, capthick=0.5, color='k',label='Koinzidenz')
ax1.plot(x,y)
ax1.set_xlabel('Schwelle des Z12 Diskriminators')
ax1.set_ylabel('Anzahl der Koinzidenten Ereignisse')
#plt.savefig('pics/Schwellenkurve.png')
plt.show()
