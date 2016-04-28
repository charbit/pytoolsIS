# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 08:25:20 2016

@author: maurice

Synopsis:
    - Perform signals with LOC for given randomness of the 
      slowness vector using the function "synthetizer"
    - Perform the MSC using the function "estimSCP"

"""
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)


import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')

from geoloc import extractstationlocations


from toolIS import maxfstat, synthetizer, generateISwithLOCgaussian, estimSCP
from toolIS import evalCRBwithgaussianLOC, CRBonazimuthonlywithoutLOC, logMSCtheoGaussian

# attention we only test the LOC
# therefore the other values are set at NaN

from numpy import array, ones, size, mean, zeros, dot, pi, nanstd
from numpy import diag, nan, max, std, exp, conjugate, fix
from numpy import random, cos, sin, sqrt, arange
from numpy import ceil, log2, real, concatenate
from scipy.linalg import sqrtm, norm
from scipy import fft, ifft , any, isreal, argsort, isnan
from scipy.signal import lfilter, butter
from scipy.signal.spectral import coherence

from matplotlib import  pyplot as plt

#listIS = ('I02','I04','I05','I06','I07','I08','I09',
#     'I10','I11','I13','I14','I17','I18','I19','I21',
#     'I23','I24','I26','I27','I30','I31','I32','I33',
#     'I34','I35','I36','I37','I39','I40','I41','I42',
#     'I43','I44','I45','I46','I47','I48','I49','I50',
#     'I51','I52','I53','I55','I56','I57','I58','I59')

#listIS = ('I27','I30','I31','I32','I33',
#     'I34','I35','I36','I37','I39','I40','I41','I42',
#     'I43','I44','I45','I46','I47','I48','I49','I50',
#     'I51','I52','I53','I55','I56','I57','I58','I59')
#
#02:M=5
#22:M=4
#27:M=18
#30:M=6
#31:M=8
#32:M=7
#33:M=8 (with a few co-located)
#34,M=12 (with a few co-located)
#45:M=4


#=================

#listIS = ('I22',)

# from scipy.signal import lfilter, butter

station = 'I22'
sensors = extractstationlocations(station, ReferenceEllipsoid=23)
nbsensors = len(sensors)
xsensors_m = zeros([nbsensors,3])
if nbsensors == 0:
    print 'no sensor in %s'%station
else:
    for im in range(nbsensors):
        evi   = sensors[im]
        xsensors_m[im,:] = array([evi.geolocs.X_km,
                       evi.geolocs.Y_km, 
                       evi.geolocs.Z_km])*1000.0;
    
    xsensors_m = xsensors_m - xsensors_m[0,:]
    M = size(xsensors_m,0);

T_sec               = 1200;
Fs_Hz               = 20.0;
N                   = int(T_sec*Fs_Hz);

#%%
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
        distances[cp] = norm(xsensors_m[i1,:]-xsensors_m[i2,:]);
        diffloc       = xsensors_m[i2,:] - xsensors_m[i1,:];
        couples[:,cp] = array([cp,i1+1,i2+1])
        cp=cp+1;       

x = generateISwithLOCgaussian(T_sec, Fs_Hz, xsensors_m,
                              azimuth0_deg,
                              elevation0_deg,
                              velocity0_mps,
                              sigma_azimuth_deg,
                              sigma_elevation_deg,
                              sigma_velocity_mps);

MSC  = zeros([Lfft,C])
cp   = 0
for i1 in range(M-1):
    for i2 in range(i1+1,M):
        frqsFFT_Hz, SDs11, SDs22, SDs12, MSC[:,cp] = estimSCP(x[:,i1],x[:,i2],Lfft,0.5,Fs_Hz, 'hanning');
        cp=cp+1;

logMSCtheo = logMSCtheoGaussian(xsensors_m, frqsFFT_Hz, azimuth0_deg, 
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
    
#    figLOC.savefig('tt.pdf', facecolor = [1,1,0.92], format='pdf')
    #%%    
#plt.figure(C)
#for im in range(M):
#    plt.plot(xsensor_m[im,0],xsensor_m[im,1],'x')
#    plt.text(xsensor_m[im,0],xsensor_m[im,1],im+1)
