# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 07:35:43 2016

@author: maurice
"""

"""
Created on Mon Feb 15 18:26:22 2016

@author: maurice
"""


import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolutilities')

from toolboxInfrasonic import MCCM, consistence
from geoloc import extractstationlocations

from toolboxInfrasonic import estimSCP, generateISsto, logMSCtheoGaussian


class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
from matplotlib import  pyplot as plt
from numpy import array, size, zeros, exp



#============================================================
station = 3;
if station == 1:
    numselect = 'I31'
    sensors = extractstationlocations(numselect, ReferenceEllipsoid=23)
    nbsensors = len(sensors)
    
    xsensors_m = zeros([nbsensors,3])

    for im in range(nbsensors):
        evi   = sensors[im]
        xsensors_m[im,:]=array([evi.geolocs.X_km,evi.geolocs.Y_km, evi.geolocs.Z_km])*1000.0;
    
    xsensors_m = xsensors_m - xsensors_m[0,:]
#        #=== I31KZ
#        xsensors_m =  1000*array([
#            [-0.05997213,  0.194591122,   0.3911],
#            [0.229169719,   0.083396195,   0.3921],
#            [0.122158887,  -0.206822564,   0.3918],
#            [-0.12375342,  -0.087843992,   0.3902],
#            [-0.026664123,   0.015567290,   0.391],
#            [0.919425013,   0.719431175,   0.3924],
#            [0.183105453,  -1.103053672,   0.3831],
#            [-1.2434694,   0.384734446,   0.3976]])
if station == 2:
        #=== I22
        xsensors_m = 1.0e+03 * array([
            [-0.088034341864435,  -0.095905624230955,   0.272],
            [-0.217769161454130,   1.227314002838975,   0.240],
            [1.046630508831105,  -0.508438802082452,    0.283],
            [-0.740827005512541,  -0.622969576526358,   0.246]])
if station == 3:
        #=== I22
        xsensors_m = 1.0e+03 *array([
            [-0.088034341864435,  -0.095905624230955,   0.272],
            [-0.217769161454130,   1.227314002838975,   0.240],
            [1.046630508831105,  -0.508438802082452,    0.283]])

#=============================================
#
M = size(xsensors_m,0);
Fs_Hz = 20.0;
Lfft = 512;
overlapFFT = 0.5;
smoothwindow = 'hanning'

azimuth_deg = 30.0;
elevation_deg = 15.0;
velocity_mps = 340.0;
sigma_azimuth_deg = 1.0;
sigma_elevation_deg = 1.0;
sigma_velocity_mps = 15.0;

N = 30000;
x, tau_pts = generateISsto(N, Fs_Hz, xsensors_m, azimuth_deg, 
                  elevation_deg, velocity_mps, sigma_azimuth_deg, 
                  sigma_elevation_deg, sigma_velocity_mps, nbRays=100);

HorizontalSize = 8
VerticalSize   = 20

fig = plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);

combi = M*(M-1)/2
MSC = zeros([Lfft,combi])
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
        plt.plot(frqsFFT_Hz, exp(logMSC_theo[:,cp]))
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