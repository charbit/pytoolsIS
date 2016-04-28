# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:53:06 2016

Compute the ROC curve associated to the FSTAT
for no LOC and LOC signals
Under H0 we have only noise
Under H1 we use the function "synthetizer"
     without and with LOC
     the true azimut is randomly chosen
The maximization on Fstat is only w.r.t. the azimuth over
the range "range_azimuth_deg"

@author: maurice

Synopsis:
This program performs the ROC curves for the Fstat function
of test without/with LOC


"""
import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolbenchmark')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolboxIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolutilities')

from toolIS import *
from geoloc import extractstationlocations

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

# attention we only test the LOC
# therefore the other values are set at NaN

from matplotlib import  pyplot as plt
from numpy import array, ones, size, mean, zeros, dot, pi, fix
from numpy import diag, nan, max, std, exp, conjugate
from numpy import random, cos, sin, sqrt
from numpy import ceil, log2, real, concatenate
from scipy.linalg import sqrtm
from scipy import fft, ifft , any, isreal, argsort, isnan

from scipy.signal import lfilter, butter

from scipy.signal.spectral import coherence

from geoloc import *

#============================================================
#Lruns0 = 200
#Lruns1 = 200
#flag_LOC = 1;
#SOISNRrange_dB = array([-10.0, -10.0]);
#LOC_std_azimuth_deg         = 5.0;
#LOC_std_elevation_deg       = 3.0;
#LOC_std_velocity_mps        = 10.0;

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
                   
nbsensors = size(xsensors_m,0)        

flag_SON                    = 0;
flag_hugeNoise              = 0;
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
SOIdurationrange_s          = array([30.0, 30.0]);
SOIfrequencywidth_Hz        = array([0.03, 3.5]);

# if SOI_margeazumith_deg greater 
#    than SOIazimuthrange_deg[1]-SOIazimuthrange_deg[0]
# SOI_margeazumith_deg has no effect
SOI_margeazumith_deg         = 5;
SOIelevationrange_deg        = SOIelevation0_deg * array([1, 1]);
SOIvelocityrange_mps         = SOIvelocity0_mps * array([1, 1]);
SOIduration_at_least_sec     = 0;
SOI_margeazumith_deg         = 10;
totalTime_s                  = 200.0;

range_azimuth_deg   = linspace(0.0,359.0,100) #array([50.0]) ; #
range_elevation_deg = array([SOIelevationrange_deg[0]]);#array([10.0])
range_velocity_mps  = array([SOIvelocityrange_mps[0]]);# array([340.0])


valH0  = zeros(Lruns0)
valH1  = zeros(Lruns1)
N      = int(SOIdurationrange_s[0]*SOIFs_Hz)

for i0 in range(Lruns0):
    x0 = random.randn(N,nbsensors)
    aux0 = maxfstat(x0,SOIFs_Hz,xsensors_m, range_azimuth_deg, 
             range_elevation_deg,range_velocity_mps)
    valH0[i0] = aux0[0]
        
for i1 in range(Lruns1):
    SOIazimuthrange_deg = fix(random.rand()*360.0)*array([1.0, 1.0]);
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
             margeazumith_deg = SOI_margeazumith_deg,
             flag_successiveazimuth = flag_SOIsuccessiveazimuth)
             
    x1,listevents  = synthetizer(sensors, SOI, LOC, 
              SON, hugeNoise, failure, emergent);
           
    evi1 = listevents[0];
    tstart_pts = int(evi1.SOI.TOA_sec*SOIFs_Hz);
    tdur_pts = int(evi1.SOI.duration_sec*SOIFs_Hz);
    tend_pts = tstart_pts+tdur_pts
    
    x1prime = x1[tstart_pts:tend_pts,:]
    aux1 = maxfstat(x1prime,SOIFs_Hz,xsensors_m, range_azimuth_deg, 
             range_elevation_deg,range_velocity_mps)
    valH1[i1] = aux1[0]

#%%
alpha_percent = 95
a, b, CIalpha, CIbeta, eauc, std_eauc_exp, std_eauc_boot \
         = rocauc(valH0,valH1,alpha_percent,nbbins=60)
