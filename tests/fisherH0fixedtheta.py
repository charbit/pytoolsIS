# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:06:19 2016

@author: maurice
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:30:05 2013

@author: charbit
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
         
import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolutilities')

from geoloc import extractstationlocations
from toolIS import evalCRBwithgaussianLOC,maxfstat

from numpy import array,ones, exp, size, pi, arange
from numpy import zeros, dot, diag, cos, sin, sqrt, linspace

from numpy import random

from scipy.stats import f

from matplotlib import pyplot as plt
    
gridaz_deg = array(linspace(0,360,1))
gridel_deg = array(linspace(0,90,1))
gridc_mps = array(linspace(300,400,1))

Fs_Hz = 20

station = 'I31'

sensors = extractstationlocations(station, ReferenceEllipsoid=23)
nbsensors = len(sensors)
xsensors_m = zeros([nbsensors,3])
if nbsensors == 0:
    sys.exit('no sensor in %s'%station)
else:
    for im in range(nbsensors):
        evi   = sensors[im]
        xsensors_m[im,:] = array([evi.geolocs.X_km,
                  evi.geolocs.Y_km, 
                  evi.geolocs.Z_km])*1000.0;
    
    xsensors_m = xsensors_m - xsensors_m[0,:]
    M = size(xsensors_m,0);
T_sec = 30 ; 
N = int(T_sec*Fs_Hz)
Lruns = 10000
Fstat = zeros(Lruns)

for ir in range(0,Lruns):
    x = random.randn(N,M)
    F = maxfstat(x, Fs_Hz, xsensors_m, gridaz_deg,gridel_deg, 
                 gridc_mps)
    Fstat[ir] = F[0]
#%%

xtheo = linspace(0.5,1.5,100)
ytheo = f.pdf(xtheo,N,N*(M-1))

HorizontalSize = 5
VerticalSize   = 3
figsimul=plt.figure(num=2,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
figsimul.clf()
h1 = plt.hist(Fstat, normed=True, bins=30, label='histogram')
h2 = plt.plot(xtheo,ytheo,'r',linewidth=2, label='Fisher')
plt.legend(loc='best')

dirfigsave = '/Users/maurice/etudes/stephenA/propal2/figures/'
tt='%sthetafixFisher.pdf' %dirfigsave
plt.show()

figsimul.savefig(tt,format='pdf')

