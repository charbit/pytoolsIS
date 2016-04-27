# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 08:25:20 2016

@author: maurice
"""
import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolutilities')

from toolboxInfrasonic import logMSCtheoGaussian, estimSCP
from toolboxInfrasonic import generateISwithLOCgaussian
from numpy.linalg import norm
from numpy import array, argsort, zeros, size, exp
from numpy import random

from matplotlib import pyplot as plt

# from scipy.signal import lfilter, butter

station = 1;
if station == 1:
        #=== I31KZ
        xsensor_m =  1000*array([
            [-0.059972130000000,  0.194591122000000,   0.391100000000000],
            [0.229169719000000,   0.083396195000000,   0.392100000000000],
            [0.122158887000000,  -0.206822564000000,   0.391800000000000],
            [-0.123753420000000,  -0.087843992000000,   0.390200000000000],
            [-0.026664123000000,   0.015567290000000,   0.391000000000000],
            [0.919425013000000,   0.719431175000000,   0.392400000000000],
            [0.183105453000000,  -1.103053672000000,   0.383100000000000],
            [-1.243469400000000,   0.384734446000000,   0.397600000000000]])
if station == 2:
        #=== I22
        xsensor_m = 1.0e+03 * array([
            [-0.088034341864435,  -0.095905624230955,   0.272000000000000],
            [-0.217769161454130,   1.227314002838975,   0.240000000000000],
            [1.046630508831105,  -0.508438802082452,    0.283000000000000],
            [-0.740827005512541,  -0.622969576526358,   0.246000000000000]])
if station == 3:
        #=== I22
        xsensor_m = 1.0e+03 *array([
            [-0.088034341864435,  -0.095905624230955,   0.272000000000000],
            [-0.217769161454130,   1.227314002838975,   0.240000000000000],
            [1.046630508831105,  -0.508438802082452,    0.283000000000000]])

T_sec               = 1200;
Fs_Hz               = 20.0;
M                   = size(xsensor_m,0);
N                   = int(T_sec*Fs_Hz);

azimuth0_deg   = 360*random.rand();
elevation0_deg = 40+30*random.rand();
velocity0_mps  = 340;

sigma_azimuth_deg    = 3;
sigma_elevation_deg  = 5;
sigma_velocity_mps   = 10;

x         = zeros([N,M]);

Lfft      = 512

Lruns     = 200;
C         = M*(M-1)/2;
distances = zeros(C)
couples   = zeros([3,C])

cp        = 0;
for i1 in range(M-1):
    for i2 in range(i1+1,M):
        distances[cp] = norm(xsensor_m[i1,:]-xsensor_m[i2,:]);
        diffloc       = xsensor_m[i2,:] - xsensor_m[i1,:];
        couples[:,cp] = array([cp,i1+1,i2+1])
        cp=cp+1;       

x = generateISwithLOCgaussian(T_sec, Fs_Hz, xsensor_m,
                              azimuth0_deg,
                              elevation0_deg,
                              velocity0_mps,
                              sigma_azimuth_deg,
                              sigma_elevation_deg,
                              sigma_velocity_mps)

MSC  = zeros([Lfft,C])
cp   = 0
for i1 in range(M-1):
    for i2 in range(i1+1,M):
        frqsFFT_Hz, SDs11, SDs22, SDs12, MSC[:,cp] = estimSCP(x[:,i1],x[:,i2],Lfft,0.5,Fs_Hz, 'hanning');
        cp=cp+1;

logMSCtheo = logMSCtheoGaussian(xsensor_m, frqsFFT_Hz, azimuth0_deg, 
                                elevation0_deg, velocity0_mps, sigma_azimuth_deg, sigma_elevation_deg, sigma_velocity_mps)

#%%
if 1:
    HorizontalSize = 12
    VerticalSize   = 34

    figLOC=plt.figure(num=1, figsize=(HorizontalSize,VerticalSize))
    idsort = argsort(distances)
    
    for ic in range(C):
        plt.subplot(7,4,ic+1)
        indic = idsort[ic]
        plt.plot(frqsFFT_Hz, MSC[:,indic])
        plt.plot(frqsFFT_Hz, exp(logMSCtheo[:,indic]),'r')
        plt.xlim(0,10)
        plt.ylim(0,1.1)
        plt.title('%4.1f km\n%4.1f$^{\mathrm{o}}$,%4.1f$^{\mathrm{o}}$,%4.1fm/s' %(distances[indic]/1000, azimuth0_deg, elevation0_deg, velocity0_mps))
    
    figLOC.savefig('tt.pdf', facecolor = [1,1,0.92], format='pdf')
    #%%    
#plt.figure(C)
#for im in range(M):
#    plt.plot(xsensor_m[im,0],xsensor_m[im,1],'x')
#    plt.text(xsensor_m[im,0],xsensor_m[im,1],im+1)
