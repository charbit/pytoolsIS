# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:45:55 2016

@author: maurice
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolutilities')

from geoloc import extractstationlocations

from toolIS import maxfstat, synthetizer
from toolIS import evalCRBwithgaussianLOC, CRBonazimuthonlywithoutLOC

# attention we only test the LOC
# therefore the other values are set at NaN

from numpy import array, ones, size, mean, zeros, dot, pi, nanstd
from numpy import diag, nan, max, std, exp, conjugate, fix
from numpy import random, cos, sin, sqrt, arange
from numpy import ceil, log2, real, concatenate
from scipy.linalg import sqrtm
from scipy import fft, ifft , any, isreal, argsort, isnan
from scipy.signal import lfilter, butter
from scipy.signal.spectral import coherence

from matplotlib import  pyplot as plt

listIS = ('I02','I04','I05','I06','I07','I08','I09',
     'I10','I11','I13','I14','I17','I18','I19','I21',
     'I23','I24','I26','I27','I30','I31','I32','I33',
     'I34','I35','I36','I37','I39','I40','I41','I42',
     'I43','I44','I45','I46','I47','I48','I49','I50',
     'I51','I52','I53','I55','I56','I57','I58','I59')

listISdis=list()
llistIS = len(listIS)
for istation in range(llistIS):
    station = listIS[istation]
    sensors = extractstationlocations(station, ReferenceEllipsoid=23)
    M = len(sensors)
    xsensors_m = zeros([M,3])
    
    print 'Station %s with %i sensors'%(station,M)
    distance = zeros([M*(M-1)/2,4])
    if M == 0:
        print 'no sensor in %s'%station
    else:
        for im in range(M):
            evi   = sensors[im]
            xsensors_m[im,:] = array([evi.geolocs.X_km,
                           evi.geolocs.Y_km, 
                           evi.geolocs.Z_km])*1000.0;
        
        xsensors_m = xsensors_m - xsensors_m[0,:]
    
    cp = 0
    for im in range(M-1):
        for imp in range(im+1,M):
            vaux = xsensors_m[im,:]-xsensors_m[imp,:]
            distance[cp,0] = sqrt(dot(vaux.reshape(1,3), vaux.reshape(3,1)))
            distance[cp,1] = im
            distance[cp,2] = imp
            if distance[cp,0]<10:
                distance[cp,3] = '1'
                print '\tsensor pair colocated:(%i,%i)'%(im,imp)                
            cp=cp+1
     
    aux = struct(name=station,nbsensors=M,distance=distance)
    listISdis.append(aux)        
            
#    textsensor = 'station %s: M = %i' %(station,M)
#    print textsensor

