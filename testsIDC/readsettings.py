# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:46:50 2016

@author: maurice
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

mypaths = ('/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/testsIDC/toolbox4IDC/', \
          '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities' )
Lmypaths = len(mypaths)

import sys
for ip in range(Lmypaths):
    if mypaths[ip] not in sys.path:
        sys.path.append(mypaths[ip])
         
from geoloc import extractstationlocations

import numpy as np

xx = np.loadtxt('settings1.txt', comments='#', dtype='str',
              delimiter='\n', converters=None, skiprows=0, 
              usecols=None, unpack=False, ndmin=0)
              
lxx = len(xx)
for iell in range(lxx):
    exec(str(xx[iell]))
    
SON_frequencyband_Hz = np.array([SON_frequencyband_Hz_inf, SON_frequencyband_Hz_sup])
SON  = struct(name='soninfo', flag = flag_SON,
          SNR_dB = SON_SNR_dB, azimuth_deg = SON_azimuth_deg,
          elevation_deg = SON_elevation_deg, velocity_mps = SON_velocity_mps,
          frequencyband_Hz = SON_frequencyband_Hz)
hugeNoise = struct(name = 'hugenoiseinfo', 
           flag = flag_hugeNoise,
           SNR_dB = hugeNoise_SNR_dB,
           probability = hugeNoise_probability)

LOC = struct(name='locinfo', flag = flag_LOC,
          std_azimuth_deg = LOC_std_azimuth_deg,
          std_elevation_deg = LOC_std_elevation_deg, 
          std_velocity_mps = LOC_std_velocity_mps)
failure = struct(name='failureinfo', flag = flag_failure,
          probability = failure_probability)
          
emergent = struct(name='emergentinfo', 
          flag = flag_emergent,
          min_sec = emergent_min_sec,
          durationpercent = emergent_durationpercent)

SOIdurationrange_sec        = np.array([SOIdurationrange_sec_inf, SOIdurationrange_sec_sup]);
SOIfrequencywidth_Hz        = np.array([SOIfrequencywidth_Hz_inf, SOIfrequencywidth_Hz_sup]);
SOISNRrange_dB              = np.array([SOISNRrange_dB_inf, SOISNRrange_dB_sup]);

SOIazimuthrange_deg          = np.array([SOIazimuthrange_deg_inf, SOIazimuthrange_deg_sup]);
SOIelevationrange_deg        = np.array([SOIelevationrange_deg_inf, SOIelevationrange_deg_sup]);
SOIvelocityrange_mps         = np.array([SOIvelocityrange_mps_inf, SOIvelocityrange_mps_sup]);

SOI = struct(name = 'soiinfo', 
             soiflag_real = SOIflag_real,
             soidatabase = SOI_database,
             noiseflag_real = NOISEflag_real,
             noise_database = NOISE_database,
             totaltime_sec = totalTime_sec,
             Fs_Hz = SOIFs_Hz, 
             nb_events = SOInb_events,
             durationrange_sec = SOIdurationrange_sec,
             frequencywidth_Hz = SOIfrequencywidth_Hz,
             SNRrange_dB = SOISNRrange_dB,
             azimuthrange_deg = SOIazimuthrange_deg,
             elevationrange_deg = SOIelevationrange_deg,
             velocityrange_mps = SOIvelocityrange_mps,
             flagtimeoverlap = flag_SOItimeoverlap,
             duration_at_least_sec = SOIduration_at_least_sec,
             flag_phasedistortion = flag_SOIphasedistortion,
             margeazumith_deg   =   SOI_margeazumith_deg,
             flag_successiveazimuth = flag_SOIsuccessiveazimuth)
             
station = extractstationlocations(numselect, ReferenceEllipsoid=23)
nbsensors = len(station)
xsensors_m = np.zeros([nbsensors,3])
for im in range(nbsensors):
    evi   = station[im]
    xsensors_m[im,:] = np.array([evi.geolocs.X_km,evi.geolocs.Y_km, evi.geolocs.Z_km])*1000.0;

xsensors_m = xsensors_m - xsensors_m[0,:]
sensors = []
for im in range(np.size(xsensors_m,0)):
    geoloc_im = struct(X_km=xsensors_m[im,0]/1000.0,
                       Y_km=xsensors_m[im,1]/1000.0,
                       Z_km=xsensors_m[im,2]/1000.0)
    str_aux = struct(geolocs = geoloc_im)
    sensors.append(str_aux);




