# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:26:22 2016

@author: maurice
"""


class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)



import sys
sys.path.insert(0, './')
sys.path.insert(0, 'toolbox4IDC')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')

from geoloc import extractstationlocations
from toolidc import synthetizer
from synthetizersettings import settingson

# attention we only test the LOC
# therefore the other values are set at NaN


from numpy import array, ones, zeros, size, mean, zeros, dot, pi
from numpy import diag, nan, max, std, exp, conjugate
from numpy import random, cos, sin, sqrt
from numpy import ceil, log2, real, concatenate

from scipy.linalg import sqrtm
from scipy import fft, ifft , any, isreal, argsort, isnan
from scipy.signal import lfilter, butter
from scipy.signal.spectral import coherence
from matplotlib import  pyplot as plt

#%%
#=================== setting ===========================
# The function returns the station which is M structures
# each of them with the following fields:
#        .name = sensor name, ex. H1
#        .geolocs = structure which contains 5 items:
#           latitude_deg,longitude_deg, X_km,Y_km, Z_km
#===
station, SOI, LOC, SON, hugeNoise, failure, \
     emergent, xsensors_m = settingson()
#=================== synthetize =========================
oo, listevents  = synthetizer(station, SOI, LOC, \
                SON, hugeNoise, failure, emergent);             
#%%
SOIFs_Hz = SOI.Fs_Hz
M=size(xsensors_m,0)
t_sec = array(range(size(oo,0)))/SOIFs_Hz
t_hour = t_sec/3600.0
CM = M*(M-1)/2
distance = zeros(CM)
indexdistance = zeros([CM,2])
cp = 0;
for im1 in range(M-1):
    for im2 in range(im1+1,M):
        diffxsensors = xsensors_m[im1,:]-xsensors_m[im2,:];
        distance[cp] = sqrt(dot(diffxsensors,diffxsensors))
        indexdistance[cp,:] = array([im1,im2]);
        cp=cp+1
        
argsortdistance = argsort(distance)
indexdistance   = indexdistance[argsortdistance]
tmps=array(range(size(oo,0)))/SOIFs_Hz;

HorizontalSize = 5
VerticalSize   =10

figpdfsoi = plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
for im in range(M):
    plt.subplot(M,1,im+1)
    plt.plot(tmps,oo[:,im])
    plt.xticks(fontsize=10)
    plt.yticks([],fontsize=10)
plt.xlabel('time - s')
plt.show()
plt.hold('off')


