# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:01:52 2016

@author: maurice
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
         
import sys
sys.path.insert(0, 'toolbox4IDC')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')

from geoloc import extractstationlocations
from toolidc import synthetizer


from numpy import array, zeros, size


def settingson():
    
    """
     Settings for the call of the synthetizer function
     No input argument:
     Outputs:
      station, SOI, LOC, SON, hugeNoise, failure, emergent, xsensors_m
    """
    numselect = 'I31'
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
    SOISNRrange_dB              = array([100.0, 100.0]);
    
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
    
    #===================
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
                 flag_successiveazimuth = flag_SOIsuccessiveazimuth)
                 
    station = extractstationlocations(numselect, ReferenceEllipsoid=23)
    nbsensors = len(station)
    xsensors_m = zeros([nbsensors,3])
    for im in range(nbsensors):
        evi   = station[im]
        xsensors_m[im,:]=array([evi.geolocs.X_km,evi.geolocs.Y_km, evi.geolocs.Z_km])*1000.0;
    
    xsensors_m = xsensors_m - xsensors_m[0,:]
    sensors = []
    for im in range(size(xsensors_m,0)):
        geoloc_im = struct(X_km=xsensors_m[im,0]/1000.0,
                           Y_km=xsensors_m[im,1]/1000.0,
                           Z_km=xsensors_m[im,2]/1000.0)
        str_aux = struct(geolocs = geoloc_im)
        sensors.append(str_aux);
    
    return station, SOI, LOC, SON, hugeNoise, failure, emergent, xsensors_m
