#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:22:46 2019

@author: mmbon
"""

import numpy as np
import sympy as sy
import math
from lmfit import Model as md
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy
from scipy.odr import RealData,Model,ODR

def geradenfit(x, y, y_err):
    x = x
    y = y
    delta_y = y_err

    y_s1 =[]

    for i in range(len(y)):
        y_s1.append(1/((delta_y[i])**2))

    y_s = np.sum(y_s1)
    #print('y_s =', y_s)
    sigma_y2 = len(y)/y_s
    x_m1 = 0
    y_m1 = 0
    xy_m1 = 0
    xx_m1 = 0

    for j in range(len(x)):
        x_m_help = x[j]*(1/((delta_y[j])**2))
        x_m1 = x_m1 + x_m_help
        y_m_help = y[j]*(1/((delta_y[j])**2))
        y_m1 = y_m1 + y_m_help
        xy_m_help = (x[j]*y[j])*(1/((delta_y[j])**2))
        xy_m1 = xy_m1 + xy_m_help
        xx_m_help = (x[j]*x[j])*(1/((delta_y[j])**2))
        xx_m1 = xx_m1 + xx_m_help

    x_m = x_m1/y_s
    y_m = y_m1/y_s
    xy_m = xy_m1/y_s
    xx_m = xx_m1/y_s

    m_err = np.sqrt(sigma_y2/(len(y)*(xx_m-x_m**2)))
    n_err = m_err*np.sqrt(xx_m)

    #print('x_m =', x_m, 'y_m =', y_m, 'xy_m =', xy_m, 'xx_m =', xx_m)
    m = (xy_m - (x_m*y_m))/(xx_m - ((x_m)**2))
    n = y_m -m*x_m
    #print('m =', m, 'n =', n)
    #print('N =', len(x))

    return(m,n,m_err,n_err)


def fehlerfortpflanzung4(function, values, error_values):
    x,y,z,m = sy.symbols("x y z m")
    f = sy.sympify(function)

    f_x = (sy.diff(f,x))**2
    f_x_val = f_x.subs([(x,values[0]),(y,values[1]),(z,values[2]),(m,values[3])])
    f_y = (sy.diff(f,y))**2
    f_y_val = f_y.subs([(x,values[0]),(y,values[1]),(z,values[2]),(m,values[3])])
    f_z = (sy.diff(f,z))**2
    f_z_val = f_z.subs([(x,values[0]),(y,values[1]),(z,values[2]),(m,values[3])])
    f_m = (sy.diff(f,m))**2
    f_m_val = f_m.subs([(x,values[0]),(y,values[1]),(z,values[2]),(m,values[3])])

    result = math.sqrt((f_x_val*(error_values[0])**2)+(f_y_val*(error_values[1])**2)+(f_z_val*(error_values[2])**2)+(f_m_val*(error_values[3])**2))

    return result

def fehlerfortpflanzung3(function, values, error_values):
    x,y,z = sy.symbols("x y z")
    f = sy.sympify(function)

    f_x = (sy.diff(f,x))**2
    f_x_val = f_x.subs([(x,values[0]),(y,values[1]),(z,values[2])])
    f_y = (sy.diff(f,y))**2
    f_y_val = f_y.subs([(x,values[0]),(y,values[1]),(z,values[2])])
    f_z = (sy.diff(f,z))**2
    f_z_val = f_z.subs([(x,values[0]),(y,values[1]),(z,values[2])])
    result = math.sqrt((f_x_val*(error_values[0])**2)+(f_y_val*(error_values[1])**2)+(f_z_val*(error_values[2])**2))

    return result

def fehlerfortpflanzung2(function, values, error_values):
    x,y = sy.symbols("x y")
    f = sy.sympify(function)

    f_x = (sy.diff(f,x))**2
    f_x_val = f_x.subs([(x,values[0]),(y,values[1])])
    f_y = (sy.diff(f,y))**2
    f_y_val = f_y.subs([(x,values[0]),(y,values[1])])


    result = math.sqrt((f_x_val*(error_values[0])**2)+(f_y_val*(error_values[1])**2))

    return result

def fehlerfortpflanzung1(function, values, error_values):
    x = sy.symbols("x")
    f = sy.sympify(function)

    f_x = (sy.diff(f,x))**2
    f_x_val = f_x.subs([(x,values[0])])

    result = math.sqrt((f_x_val*(error_values[0])**2))

    return result

def gaus1(x,A1,mu1,sig1,n):
    return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+n

def gaus2(x,A1,mu1,sig1,A2,mu2,sig2,n):
    return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+A2*np.exp(-((x-mu2)**2)/(2*sig2**2))+n

def gaus3(x,A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,n):
    return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+A2*np.exp(-((x-mu2)**2)/(2*sig2**2))+A3*np.exp(-((x-mu3)**2)/(2*sig3**2))+n

def gaus1_lin(x,A1,mu1,sig1,m,n):
    return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+m*x+n

def gaus2_lin(x,A1,mu1,sig1,A2,mu2,sig2,m,n):
    return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+A2*np.exp(-((x-mu2)**2)/(2*sig2**2))+m*x+n

def gaus3_lin(x,A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,m,n):
    return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+A2*np.exp(-((x-mu2)**2)/(2*sig2**2))+A3*np.exp(-((x-mu3)**2)/(2*sig3**2))+m*x+n

class gaussfit():
    '''range als array=[lower range, upper range] angeben'''
    def __init__(self,xarray,yarray,Initial_params,min_err,yerror=None,xerror=None,gaus='1',method='lmfit', do_plot=False,interval=None):
        self.xarray_ganz = xarray
        self.yarray_ganz = yarray
        self.xerror_ganz = xerror
        self.yerror_ganz = yerror
        self.Anzahl_gauskurven = gaus
        self.method = method
        self.params = Initial_params
        self.initiate_arrays(xarray,yarray,xerror,yerror,interval,min_err)
        self.use_method(do_plot)

    def initiate_arrays(self,xarray,yarray,xerror,yerror,interval,min_err):
        if(interval is not None):
            self.xarray = xarray[interval[0]:interval[1]]
            self.yarray = yarray[interval[0]:interval[1]]
        else:
            self.xarray = xarray
            self.yarray = yarray

        if(xerror is not None and interval is not None):
            self.xerror = xerror[interval[0]:interval[1]]
            for i in range(len(self.xerror)):
                self.xerror[i] = max(self.xerror[i],min_err)
        elif(xerror is not None and interval is None):
            self.xerror = xerror
            for i in range(len(self.xerror)):
                self.xerror[i] = max(self.xerror[i],min_err)
        else:
            self.xerror=None

        if(yerror is not None and interval is not None):
            self.yerror = yerror[interval[0]:interval[1]]
            for i in range(len(self.yerror)):
                self.yerror[i] = max(self.yerror[i],min_err)
        elif(yerror is not None and interval is None):
            self.yerror = yerror
            for i in range(len(self.yerror)):
                self.yerror[i] = max(self.yerror[i],min_err)
        else:
            self.yerror=None

    def gaus_lmfit(self):
        if (self.Anzahl_gauskurven == '1'):
            self.gaus_func= gaus1
            gmod=md(self.gaus_func)
            self.params_lmfit = gmod.make_params(A1=self.params[0],mu1= self.params[1],sig1=self.params[2],n=self.params[3])

        if (self.Anzahl_gauskurven == '2'):
            self.gaus_func= gaus2
            gmod=md(self.gaus_func)
            self.params_lmfit = gmod.make_params(A1=self.params[0],mu1= self.params[1],sig1=self.params[2],A2=self.params[3],mu2= self.params[4],sig2=self.params[5],n=self.params[6])

        if (self.Anzahl_gauskurven == '3'):
            self.gaus_func= gaus3
            gmod=md(self.gaus_func)
            self.params_lmfit = gmod.make_params(A1=self.params[0],mu1= self.params[1],sig1=self.params[2],A2=self.params[3],mu2= self.params[4],sig2=self.params[5],A3=self.params[6],mu3= self.params[7],sig3=self.params[8],n=self.params[9])

        if (self.Anzahl_gauskurven == '1+lin'):
            self.gaus_func= gaus1_lin
            gmod=md(self.gaus_func)
            self.params_lmfit = gmod.make_params(A1=self.params[0],mu1= self.params[1],sig1=self.params[2],m=self.params[3],n=self.params[4])

        if (self.Anzahl_gauskurven == '2+lin'):
            self.gaus_func= gaus2_lin
            gmod=md(self.gaus_func)
            self.params_lmfit = gmod.make_params(A1=self.params[0],mu1= self.params[1],sig1=self.params[2],A2=self.params[3],mu2= self.params[4],sig2=self.params[5],m=self.params[6],n=self.params[7])

        if (self.Anzahl_gauskurven == '3+lin'):
            self.gaus_func= gaus3_lin
            gmod=md(self.gaus_func)
            self.params_lmfit = gmod.make_params(A1=self.params[0],mu1= self.params[1],sig1=self.params[2],A2=self.params[3],mu2= self.params[4],sig2=self.params[5],A3=self.params[6],mu3= self.params[7],sig3=self.params[8],m=self.params[9],n=self.params[10])


    def gaus_odr(self, p, x):
        if (self.Anzahl_gauskurven == '1'):
            A1,mu1,sig1,n = p
            return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+n

        if (self.Anzahl_gauskurven == '2'):
            A1,mu1,sig1,A2,mu2,sig2,n = p
            return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+A2*np.exp(-((x-mu2)**2)/(2*sig2**2))+n

        if (self.Anzahl_gauskurven == '3'):
            A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,n = p
            return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+A2*np.exp(-((x-mu2)**2)/(2*sig2**2))+A3*np.exp(-((x-mu3)**2)/(2*sig3**2))+n

        if (self.Anzahl_gauskurven == '1+lin'):
            A1,mu1,sig1,m,n = p
            return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+m*x+n

        if (self.Anzahl_gauskurven == '2+lin'):
            A1,mu1,sig1,A2,mu2,sig2,m,n = p
            return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+A2*np.exp(-((x-mu2)**2)/(2*sig2**2))+m*x+n

        if (self.Anzahl_gauskurven == '3+lin'):
            A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,m,n = p
            return A1*np.exp(-((x-mu1)**2)/(2*sig1**2))+A2*np.exp(-((x-mu2)**2)/(2*sig2**2))+A3*np.exp(-((x-mu3)**2)/(2*sig3**2))+m*x+n

    def use_method(self,do_plot):
        if (self.method == 'lmfit'):
            self.gaus_lmfit()
            gmod = md(self.gaus_func)
            if(self.yerror is not None):
                self.fit1=gmod.fit(self.yarray,self.params_lmfit,x=self.xarray,weights=1/self.yerror)
            else:
                self.fit1=self.gmod.fit(self.yarray,self.params_lmfit,x=self.xarray)
            self.fit_values=list(self.fit1.best_values.values())
            self.fit_values_error=np.sqrt(np.diag(self.fit1.covar))
            self.redchi = self.fit1.redchi
            if (do_plot):
                self.plot_lm()

        if (self.method == 'odr'):
            model = Model(self.gaus_odr)
            if(self.xerror is not None and self.yerror is not None):
                data = RealData(x=self.xarray, y=self.yarray, sx=self.xerror, sy=self.yerror)
            elif(self.xerror is not None and self.yerror is None):
                data = RealData(x=self.xarray, y=self.yarray, sx=self.xerror)
            elif(self.xerror is None and self.yerror is not None):
                data = RealData(x=self.xarray, y=self.yarray, sy=self.yerror)
            else:
                data = RealData(x=self.xarray, y=self.yarray)
            myodr = ODR(data,model,beta0=self.params)
            self.output = myodr.run()
            self.fit_values = self.output.beta
            self.fit_values_error = self.output.sd_beta
            self.redchi = self.output.res_var
            if (do_plot):
                self.plot_odr()


    def plot_lm(self):
        fig= plt.figure(figsize=(5.5,3.5))
        ax= fig.add_axes([0.155,0.15,0.8,0.8])
        ax.errorbar(x=self.xarray_ganz,y=self.yarray_ganz,yerr=self.yerror_ganz,fmt='o',markersize=1,color='red',label='Input')
        ax.errorbar(x=self.xarray,y=self.fit1.best_fit,fmt='o',markersize=1,color='blue',label='Fit Funktion')
        ax.legend()
        plt.show()

    def plot_odr(self):
        yplot=self.gaus_odr(self.output.beta,self.xarray)
        fig= plt.figure(figsize=(5.5,3.5))
        ax= fig.add_axes([0.155,0.15,0.8,0.8])
        if (self.yerror_ganz is not None and self.xerror_ganz is not None):
            ax.errorbar(x=self.xarray_ganz,y=self.yarray_ganz,yerr=self.yerror_ganz,xerr=self.xerror_ganz,fmt='o',markersize=0.3,capsize=0.3,elinewidth=0.3,markeredgewidth=0.3,color='red',label='Input',zorder=0)
        elif (self.yerror_ganz is None and self.xerror_ganz is not None):
            ax.errorbar(x=self.xarray_ganz,y=self.yarray_ganz,xerr=self.xerror_ganz,fmt='o',markersize=0.3,capsize=0.3,elinewidth=0.3,markeredgewidth=0.3,color='red',label='Input',zorder=0)
        elif (self.yerror_ganz is not None and self.xerror_ganz is None):
            ax.errorbar(x=self.xarray_ganz,y=self.yarray_ganz,yerr=self.yerror_ganz,fmt='o',markersize=0.3,capsize=0.3,elinewidth=0.3,markeredgewidth=0.3,color='red',label='Input',zorder=0)
        else:
            ax.errorbar(x=self.xarray_ganz,y=self.yarray_ganz,fmt='o',markersize=0.3,capsize=0.3,elinewidth=0.3,markeredgewidth=0.3,color='red',label='Input',zorder=0)
        ax.errorbar(x=self.xarray,y=yplot,fmt='o',markersize=1,color='blue',label='Fit Funktion',zorder=1)
        ax.legend()
        plt.show()



