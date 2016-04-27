# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:26:22 2016

@author: maurice
"""

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolutilities')

from geoloc import extractstationlocations

from toolIS import *
from geoloc import extractstationlocations

from toolIS import maxfstat, synthetizer
from toolIS import evalCRBwithgaussianLOC


class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

# attention we only test the LOC
# therefore the other values are set at NaN

from matplotlib import  pyplot as plt
from numpy import array, ones, size, mean, zeros, dot, pi
from numpy import diag, nan, max, std, exp, conjugate
from numpy import random, cos, sin, sqrt
from numpy import ceil, log2, real, concatenate
from scipy.linalg import sqrtm
from scipy import fft, ifft , any, isreal, argsort, isnan

from scipy.signal import lfilter, butter

from scipy.signal.spectral import coherence


#============================================================
station = 1;
if station == 1:
    numselect = 'I31'
    sensors = extractstationlocations(numselect, ReferenceEllipsoid=23)
    nbsensors = len(sensors)
    
    xsensors_m = zeros([nbsensors,3])

    for im in range(nbsensors):
        evi   = sensors[im]
        xsensors_m[im,:]=array([evi.geolocs.X_km,evi.geolocs.Y_km, evi.geolocs.Z_km])*1000.0;
    
    xsensors_m = xsensors_m - xsensors_m[0,:]

if station == 2:
        #=== I22
        xsensors_m = 1.0e+03 * array([
            [-0.088034341864435,  -0.095905624230955,   0.272],
            [-0.217769161454130,   1.227314002838975,   0.240],
            [1.046630508831105,  -0.508438802082452,    0.283],
            [-0.740827005512541,  -0.622969576526358,   0.246]])
        sensors = []
        for im in range(size(xsensors_m,0)):
            geoloc_im = struct(X_km=xsensors_m[im,0]/1000.0,
                               Y_km=xsensors_m[im,1]/1000.0,
                               Z_km=xsensors_m[im,2]/1000.0)
            str_aux = struct(geolocs = geoloc_im)
            sensors.append(str_aux);
                   
if station == 3:
        #=== I22
        xsensors_m = 1.0e+03 *array([
            [-0.088034341864435,  -0.095905624230955,   0.272],
            [-0.217769161454130,   1.227314002838975,   0.240],
            [1.046630508831105,  -0.508438802082452,    0.283]])
        sensors = []
        for im in range(size(xsensors_m,0)):
            geoloc_im = struct(X_km=xsensors_m[im,0]/1000.0,
                               Y_km=xsensors_m[im,1]/1000.0,
                               Z_km=xsensors_m[im,2]/1000.0)
            str_aux = struct(geolocs = geoloc_im)
            sensors.append(str_aux);

#geolocs = [];
#for im in range(M):
#    geoloc_im = struct(latitude_deg=0, longitude_deg=0, \
#    elevation_km = xsensors_m[im,2]/1000.0, \
#    X_km = xsensors_m[im,0]/1000.0, \
#    Y_km = xsensors_m[im,1]/1000.0)
#    geolocs.append(geoloc_im)
#    
#xsensors = struct(name='is31', data = geolocs);

#%%
#=============================================
#
M = size(xsensors_m,0);
flag_SON                    = 0;
flag_hugeNoise              = 0;
flag_LOC                    = 0;
flag_SOItrue                = 0;
flag_emergent               = 0;
flag_failure                = 0;
flag_SOIphasedistortion     = 0;
flag_SOItimeoverlap         = 0;
flag_SOIsuccessiveazimuth   = 1;

SON_SNR_dB                  = 100.0;
SON_azimuth_deg             = 168.0;
SON_elevation_deg           = 0.0;
SON_velocity_mps            = 340.0;
SON_frequencyband_Hz        = array([0.08, 1]);

SON  = struct(name='soninfo', flag = flag_SON,
              SNR_dB = SON_SNR_dB, azimuth_deg = SON_azimuth_deg,
              elevation_deg = SON_elevation_deg, velocity_mps = SON_velocity_mps,
              frequencyband_Hz = SON_frequencyband_Hz)
#=======================================
hugeNoise_SNR_dB            = 30;
hugeNoise_probability       = 0.05;
hugeNoise = struct(name = 'hugenoiseinfo', 
                   flag = flag_hugeNoise,
                   SNR_dB = hugeNoise_SNR_dB,
                   probability = hugeNoise_probability)
#=======================================
LOC_std_azimuth_deg         = 1.0;
LOC_std_elevation_deg       = 1.0;
LOC_std_velocity_mps        = 15.0;
LOC  = struct(name='locinfo', flag = flag_LOC,
              std_azimuth_deg = LOC_std_azimuth_deg,
              std_elevation_deg = LOC_std_elevation_deg, 
              std_velocity_mps = LOC_std_velocity_mps)

#=======================================
failure_probability         = 0.05;
failure = struct(name='failureinfo', flag = flag_failure,
                 probability = failure_probability)

#=======================================
emergent_min_sec            = 20;
emergent_durationpercent    = 50;
emergent = struct(name='emergentinfo', 
                  flag = flag_emergent,
                  min_sec = emergent_min_sec,
                  durationpercent = emergent_durationpercent)

#=======================================
#============ signal of interest (SOI)
SOI_database                = '../I48TN/I48.dat';
NOISEflag_real              = 0;
NOISE_database              = '../I48TN/I48.dat';
SOIFs_Hz                    = 20.0;

SOInb_events                = 1;
SOIdurationrange_s          = array([20.0, 20.0]);
SOIfrequencywidth_Hz        = array([0.03, 3.5]);
SOISNRrange_dB              = array([0.0, 0.0]);

SOIazimuthrange_deg          = array([50.0, 50.0]);
# if SOI_margeazumith_deg greater 
#    than SOIazimuthrange_deg[1]-SOIazimuthrange_deg[0]
# SOI_margeazumith_deg has no effect
SOI_margeazumith_deg         = 5 

SOIelevationrange_deg        = array([20.0, 20.0]);
SOIvelocityrange_mps         = array([340.0, 340.0]);

SOIduration_at_least_sec    = 0

SOI_margeazumith_deg        = 10 


totalTime_s                 = 100.0;

SOI = struct(name = 'soiinfo', flag_SOItrue = flag_SOItrue,
             soidatabase = SOI_database,
             noiseflag_real = NOISEflag_real,
             noise_database = NOISE_database,
             totaltime_s = totalTime_s,
             Fs_Hz = SOIFs_Hz, 
             nb_events = SOInb_events,
             durationrange_s = SOIdurationrange_s,
             frequencywidth_Hz = SOIfrequencywidth_Hz,
             SNRrange_dB = SOISNRrange_dB,
             azimuthrange_deg = SOIazimuthrange_deg,
             elevationrange_deg = SOIelevationrange_deg,
             velocityrange_mps = SOIvelocityrange_mps,
             flagtimeoverlap = flag_SOItimeoverlap,
             duration_at_least_sec = SOIduration_at_least_sec,
             flag_phasedistortion = flag_SOIphasedistortion,
             margeazumith_deg   =   SOI_margeazumith_deg,
             flag_successiveazimuth = flag_SOIsuccessiveazimuth
)


#%%
#    
#    print('******************* generation *****************');
#    #==== computation

oo,listevents  = synthetizer(sensors, SOI, LOC, 
                             SON, hugeNoise, failure, emergent);
# 
t_sec = array(range(size(oo,0)))/SOIFs_Hz
t_hour = t_sec/3600.0
#%%


#for im in range(M):
#    plt.subplot(M,1,im+1)
#    plt.plot(t_hour,oo[:,im])
#    plt.plot(ev0.SOITOA_sec/3600.0+ev0.SOIdelaywrtTOA_sec[im]/3600.0,0,'or')
#    plt.xlim(ev0.SOITOA_sec/3600.0+[-130.0/3600.0,130.0/3600.0])

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


#%%
HorizontalSize = 8
VerticalSize   = 20

figpdfsoi = plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);

Lfft = 512;
frqs_Hz = array(range(Lfft))*SOIFs_Hz/Lfft;
frqs_Hz2 = frqs_Hz **2;


cp = 0;
for ik in range(SOInb_events):
    ev_ik=listevents[ik];
    logMSCtheobis = logMSCtheoGaussian(xsensors_m, 
                   frqs_Hz,
                   ev_ik.SOI.azimuth_deg,
                   ev_ik.SOI.elevation_deg,
                   ev_ik.SOI.velocity_mps,
                   ev_ik.LOC.std_azimuth_deg,
                   ev_ik.LOC.std_elevation_deg,
                   ev_ik.LOC.std_velocity_mps)
                   
    MSCtheo = zeros([CM,Lfft])
    if not(isnan(any(ev_ik.LOC.Gamma2_epsilon))):
        Gamma2_epsilon = ev_ik.LOC.Gamma2_epsilon;
        cp = 0;
        for im1 in range(M-1):
            for im2 in range(im1+1,M):
                sumGamma2_epsilon = dot(dot((xsensors_m[im1,:]-xsensors_m[im2,:]),Gamma2_epsilon), 
                                        conjugate(xsensors_m[im1,:]-xsensors_m[im2,:]))              
                for idf in range(Lfft):
                    MSCtheo[cp,idf] = exp(-4*pi*pi*frqs_Hz2[idf]*sumGamma2_epsilon);
                cp=cp+1

    for cp in range(CM):
        im1ord = int(indexdistance[cp,0])
        im2ord = int(indexdistance[cp,1])
        
        id00 = (ev_ik.SOI.TOA_sec+ev_ik.SOI.delayTOA_sec[im1ord])*SOIFs_Hz
        id01 = id00 + ev_ik.SOI.duration_sec*SOIFs_Hz
        id10 = max([0,(ev_ik.SOI.TOA_sec+ev_ik.SOI.delayTOA_sec[im2ord])*SOIFs_Hz])
        id11 = id10 + ev_ik.SOI.duration_sec*SOIFs_Hz
        x0   = oo[int(id00):int(id01),im1ord];
        x1   = oo[int(id10):int(id11),im2ord];
        f, Cxy = coherence(x0,x1, nfft=Lfft);
        plt.subplot(7,4,cp+1)
        plt.semilogy(f*SOIFs_Hz, Cxy)
#        plt.semilogy(frqs_Hz,MSCtheo[argsortdistance[cp],:])
#        plt.semilogy(frqs_Hz,exp(logMSCtheobis[:,argsortdistance[cp]]),'r')
        tt="d = %.0f m"% distance[argsortdistance[cp]]
        plt.title(tt)
        plt.xlim([0.0, SOIFs_Hz/2])
        plt.ylim([0.01,1])
        

    xextract = oo[int(id00):int(id01),:];
    xx=xextract[0:1999,:]
    
    xalign, tkl_pts = alignmentwrt1(xextract,1,200, visu=1);
    #plt.plot(xalign);plt.xlim([000,3000]);plt.ylim([-2,2])
#%%
#    r = 1
#    cons, tkl = consistence(xx, rate=r)
#    print cons
#    print tkl
#    print ev_ik.SOI.delaywrtTOA_sec*20-ev_ik.SOI.delaywrtTOA_sec[0]*20
#%%
#    r = 3
#    mccm = MCCM(xx, rate=r)
#    print mccm

#%%
