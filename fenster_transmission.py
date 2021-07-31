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

energy = np.loadtxt("polypropylene_window_10micron.txt", skiprows=2,delimiter=',', usecols=0) * 0.001
transmission_coefficient = np.loadtxt("polypropylene_window_10micron.txt", skiprows=2,delimiter=',', usecols=1)
transmission_coefficient_err = np.sqrt(transmission_coefficient) + 1e-60

# energy_gaus = energy[:36]
# trans_gaus = transmission_coefficient[:36]
# trans_gaus_err = transmission_coefficient_err[:36]
#
# energy_log = energy[18:]
# trans_log = transmission_coefficient[18:]
# trans_log_err = transmission_coefficient_err[18:]
#
# def fitting_gaus(x,amp_gaus,x0_gaus,sig_gaus):
#     return amp_gaus*np.exp(-0.5*((x-x0_gaus)/sig_gaus)**2)
#
#
# def fitting_logistic_func(x,amp_gaus,x0_gaus,sig_gaus,L_logis,x0_logis,k_logis,a,b):
#     hp1 = amp_gaus*np.exp(-0.5*((x-x0_gaus)/sig_gaus)**2)
#     hp2 = L_logis/(1+np.exp(-k_logis*(x-x0_logis)))
#     hp3 = a*x+b
#     return hp1 + hp2 +hp3
#
#
# def fittin_solo_logistic(x,L_logis,x0_logis,k_logis,a,b):
#     hp2 = L_logis / (1+np.exp(-k_logis * (x-x0_logis)))
#     hp3 = a * x+b
#     return hp2+hp3
#
#
# def fitting_with_exp(x,a,b,c,d,e,f,g,h):
#     hp1 = a * np.exp(b*x+c*x**2) + d
#     hp2 = e * np.exp(f*x+g*x**2) + h
#     return 1 - hp1 - hp2
#
#
# gmod = md(fitting_gaus)
# params = gmod.make_params(amp_gaus=0.18,x0_gaus=0.27,sig_gaus=0.01)
# fit1 = gmod.fit(transmission_coefficient,params,x=energy)
# parem1 = list(fit1.best_values.values())
# parem_err1 = np.sqrt(np.diag(fit1.covar))
# print(fit1.fit_report())
#
# # gmod2 = md(fitting_logistic_func)
# # params2 = gmod2.make_params(amp_gaus=0.25,x0_gaus=0.18,sig_gaus=1,L_logis=1,x0_logis=2,k_logis=0.5,a=0,b=0)
# # fit2 = gmod2.fit(transmission_coefficient,params2,x=energy)
# # parem2 = list(fit2.best_values.values())
# # parem_err2 = np.sqrt(np.diag(fit2.covar))
# # print(fit2.fit_report())
#
# gmod3 = md(fitting_with_exp)
# params3 = gmod3.make_params(a=0.05,b=0.1,c=0.001,d=0,e=0.01,f=-0.5,g=0.01,h=0)
# fit3 = gmod3.fit(trans_log,params3,x=energy_log)
# parem3 = list(fit3.best_values.values())
# parem_err3 = np.sqrt(np.diag(fit3.covar))
# # print(fit3.fit_report())
#
# x_array1 = np.arange(0,max(energy_gaus),0.01)
# y_array1 = fitting_gaus(x_array1,*parem1)
#
# x_array3 = np.arange(min(energy_log),max(energy_log),0.01)
# y_array3 = fitting_with_exp(x_array3,*parem3)

y_interpolated = np.interp(np.arange(0,10,10/len(energy)),energy,transmission_coefficient)

fig1 = plt.figure(figsize=(12,8), dpi=80)
ax1 = fig1.add_axes([0.15,0.15,0.8,0.8])
#ax1.errorbar(x_disc_schwelle[1:],y_count_coincidence[1:],yerr=np.sqrt(y_count_coincidence[1:]), ls='none', capsize=2,elinewidth=0.5, capthick=0.5, color='k',label='Koinzidenz')
ax1.plot(energy,transmission_coefficient,color='red',label='Datenpunkte')
ax1.plot(energy,y_interpolated,color='green',label='Intrerpoliert')
ax1.set_xlabel('Energie in keV')
ax1.set_ylabel('Wahrscheinlichkeit der Transmission durch des pp-Fenster')
ax1.legend(loc='upper left')
# plt.savefig('pics/Schwellenkurve.png')
plt.show()