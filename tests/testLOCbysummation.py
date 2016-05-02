# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 07:35:43 2016

@author: maurice

Generate LOC by summing multi-rays with slowness dispersion
The MSC is performed and compred to the theoretical
curve

"""

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolcalibration')

from geoloc import extractstationlocations

from toolIS import MCCM, consistence
from toolIS import estimSCP, generateISsto, logMSCtheoGaussian

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
from matplotlib import  pyplot as plt
from numpy import array, size, zeros, exp

station = 'I31'
sensors = extractstationlocations(station, 
                    ReferenceEllipsoid=23)
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
#
M = size(xsensors_m,0);
Fs_Hz = 20.0;
Lfft = 512;
overlapFFT = 0.5;
smoothwindow = 'hanning'

azimuth_deg = 30.0;
elevation_deg = 15.0;
velocity_mps = 340.0;
sigma_azimuth_deg = 5.0;
sigma_elevation_deg = 3.0;
sigma_velocity_mps = 13.0;

N = 10000;
x, tau_pts = generateISsto(N, Fs_Hz, xsensors_m, azimuth_deg, 
                  elevation_deg, velocity_mps, sigma_azimuth_deg, 
                  sigma_elevation_deg, sigma_velocity_mps, nbRays=30);

HorizontalSize = 3
VerticalSize   = 3

fig = plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);

plt.clf()
combi = M*(M-1)/2
MSC = zeros([Lfft,combi])
#%%
cp=0
for im1 in range(M-1):
    x1 = x[:,im1]
    for im2 in range(im1+1,M):
        x2 = x[:,im2]
        frqsFFT_Hz, SDs11, SDs22, SDs12, MSC_cp= \
              estimSCP(x1, x2, Lfft, overlapFFT, Fs_Hz, smoothwindow)
        logMSC_theo = logMSCtheoGaussian(xsensors_m, 
           frqsFFT_Hz,
           azimuth_deg,
           elevation_deg,
           velocity_mps,
           sigma_azimuth_deg,
           sigma_elevation_deg,
           sigma_velocity_mps)
           
        plt.subplot(7,4,cp+1)
        plt.semilogy(frqsFFT_Hz, MSC_cp)
        plt.hold('True')
        plt.semilogy(frqsFFT_Hz, exp(logMSC_theo[:,cp]))
        plt.xlim([0., 5.0])
        plt.ylim([0.01, 1])
        MSC[:,cp] = MSC_cp
        plt.grid()
        cp=cp+1
#%%
#print tau_pts
#for im in range(M):
#    plt.subplot(M,1,im+1)
#    plt.plot(x[:,im])