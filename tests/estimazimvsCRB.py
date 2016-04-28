# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:25:11 2016

@author: maurice
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:26:22 2016

@author: maurice
#=================================
Synopsis:
This program performs the theoretical STD from the CRB and compare to the 
experimental STD obtianed by simulation. It could be called as a function by
the program "multirunsestimAZvsCRB.py". In this case the estimation is only 
on the azimuth, therefore the CRB must be performed by the module
"CRBonazimuthonlywithoutLOC".

To know features on the stations, run first "liststations.py"

#=================================
"""
 
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)


import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')

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
#==================== init values ===========================
#llistSNR0_dB = 1
#listSNR0_dB = linspace(-5.0,-5.0,llistSNR0_dB)
#az0_deg                = 160;#random.randint(300)+20.0

#==
el0_deg                = 70.0
c0_mps                 = 340.0
dur0_sec               = 30.0;
flag_LOC               = 0;
Lruns                  = 100;

azgridscannumber       = 150;

if flag_LOC ==1:
    std_aec = struct(a_deg = 5, e_deg = 3, c_mps = 13)
else:    
    std_aec = struct(a_deg = 0, e_deg = 0, c_mps = 0)
#============================================================
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

SON = struct(name='soninfo', flag = flag_SON,
              SNR_dB = SON_SNR_dB, azimuth_deg = SON_azimuth_deg,
              elevation_deg = SON_elevation_deg, velocity_mps = SON_velocity_mps,
              frequencyband_Hz = SON_frequencyband_Hz)
#=======================================
failure_probability         = 0.05;
failure = struct(name='failureinfo', flag = flag_failure,
                 probability = failure_probability)

#=======================================
emergent_min_sec            = 20.0;
emergent_durationpercent    = 50.0;
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
#=======================================
hugeNoise_SNR_dB            = 30.0;
hugeNoise_probability       = 0.05;
hugeNoise = struct(name = 'hugenoiseinfo', 
                   flag = flag_hugeNoise,
                   SNR_dB = hugeNoise_SNR_dB,
                   probability = hugeNoise_probability)
#=======================================
              
SOIfrequencywidth_Hz        = array([0.01, 9.5]);     
# if SOI_margeazumith_deg greater 
#    than SOIazimuthrange_deg[1]-SOIazimuthrange_deg[0]
# SOI_margeazumith_deg has no effect
SOI_margeazumith_deg        = 5.0       
SOIazimuthrange_deg         = az0_deg * array([1.0, 1.0]);
SOIelevationrange_deg       = el0_deg * array([1.0, 1.0]);
SOIvelocityrange_mps        = c0_mps  * array([1.0, 1.0]);        
SOIduration_at_least_sec    = 0.0      
SOI_margeazumith_deg        = 10.0       
totalTime_s                 = 200.0;

#============================================================
LOC  = struct(name='locinfo', flag = flag_LOC,
              std_azimuth_deg = std_aec.a_deg,
              std_elevation_deg = std_aec.e_deg, 
              std_velocity_mps = std_aec.c_mps)
llistIS = len(listIS)
stdazMC_deg = zeros([llistIS, llistSNR0_dB])
stdazCRB_deg = zeros([llistIS, llistSNR0_dB])
stdazonly_CRB_deg = zeros([llistIS, llistSNR0_dB])
results = list()

for istation in range(llistIS):
    station = listIS[istation]
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
        
        SOIdurationrange_s = dur0_sec * array([1.0, 1.0]);
        
        for iSNR0_dB in range(llistSNR0_dB):
            SNR0_dB = listSNR0_dB[iSNR0_dB]
            SOISNRrange_dB = SNR0_dB * array([1.0, 1.0]);
            SOIvelocityrange_mps = c0_mps * array([1.0, 1.0]);           
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
        
            aec = struct(a_deg = az0_deg, 
                    e_deg = SOIelevationrange_deg[0],
                    c_mps = SOIvelocityrange_mps[0])

            #========== CRB calculation
            dur_sec = SOIdurationrange_s[0]        
            sigma2noise = 10.0**(-SNR0_dB/10.0)
            CRB, Jacobav_k, C0known = evalCRBwithgaussianLOC(xsensors_m, 
                    sigma2noise, aec, std_aec, dur_sec, SOIFs_Hz)
        
            STDaz_CRB_deg = sqrt(CRB.aec[0,0])*180/pi
            
            CRBazonky = CRBonazimuthonlywithoutLOC(xsensors_m, sigma2noise, aec,
                           dur_sec, SOIFs_Hz)
            stdazonly_CRB_deg[istation,iSNR0_dB] = sqrt(CRBazonky)*180/pi
    
            #=== monte-carlo simulation
            cc=4
            estimaz_deg = zeros(Lruns)
                
            amplituderange_az_deg = arange(-cc*STDaz_CRB_deg,cc*STDaz_CRB_deg,
                          2*cc*STDaz_CRB_deg/azgridscannumber);
            Optim_range_az_deg = az0_deg + amplituderange_az_deg
            Optim_range_el_deg = array([SOIelevationrange_deg[0]])
            Optim_range_vl_mps = array([SOIvelocityrange_mps[0]])
            
            for ir in range(Lruns):
                allres = synthetizer(sensors, SOI, LOC, \
                            SON, hugeNoise, failure, emergent);
                if not(len(allres)==0):
                    em0 = allres[1][0]
                    id1 = int(em0.SOI.TOA_sec*SOIFs_Hz);
                    id2 = id1 + int(em0.SOI.duration_sec*SOIFs_Hz);
                    
                    oo = allres[0]
                    xx = oo[id1:id2,:];
                    
                    res = maxfstat(xx,SOIFs_Hz,xsensors_m,
                                   Optim_range_az_deg,
                                   Optim_range_el_deg, 
                                   Optim_range_vl_mps)
                    estimaz_deg[ir]=res[1];
                    
                    if (ir%100) == 0:
                        print ir
                else:
                    estimaz_deg[ir]=nan
            
            #%%  
            aux = struct(station=station, SNR = em0.SOI.SNR_dB[0],
                         az_deg = em0.SOI.azimuth_deg, el_deg = em0.SOI.elevation_deg, 
                         c_mps = em0.SOI.velocity_mps, dur_sec = em0.SOI.duration_sec,
                         deltaaz_deg = Optim_range_az_deg[1]-Optim_range_az_deg[0], stdMC_deg = nanstd(estimaz_deg), 
                         stdCRB_deg =  STDaz_CRB_deg)
            results.append(aux)
            stdazMC_deg[istation,iSNR0_dB]=aux.stdMC_deg
            stdazCRB_deg[istation,iSNR0_dB]=aux.stdCRB_deg
 
    #%%
            ttexp = '***\nstation = %s, M = %i\nSNR = %i dB\nazimuth = %2.1f\nelevation = %2.1f\nvelocity = %2.1f m/s\nduration = %2.1f s\nAz_step_deg = %2.3f\nSTDaz_deg = %2.3f\nSTD_CRB_deg = %2.3f\nstdazonly_CRB_deg = %2.3f\n**************' \
                                         %(station, M, em0.SOI.SNR_dB[0], em0.SOI.azimuth_deg, 
                                           em0.SOI.elevation_deg, 
                                           em0.SOI.velocity_mps, em0.SOI.duration_sec, 
                                           Optim_range_az_deg[1]-Optim_range_az_deg[0], 
                                           nanstd(estimaz_deg), stdazCRB_deg[istation,iSNR0_dB], 
                                           stdazonly_CRB_deg[istation,iSNR0_dB])
            
            print ttexp

#HorizontalSize = 5
#VerticalSize   = 3
#figSTDasCRB = plt.figure(num=1,figsize=
#               (HorizontalSize,VerticalSize), edgecolor='k', 
#                facecolor = [1,1,0.92])
#
#for istation in range(llistIS):  
##    print [stdazCRB_deg[istation,:],stdazMC_deg[istation,:]]
#    plt.loglog(stdazCRB_deg,stdazMC_deg,'o')
#    plt.hold('True')
#    plt.loglog([0.01, 1],[0.01, 1],':')
#    plt.grid()
#    plt.xlabel('STD from Cramer-Rao bound')
#    plt.ylabel('STD on %i runs'%Lruns)
#    if llistIS == 1:
#        tt = 'STD by simulation vs CRB-STD\nfor station %s of the IMS (M = %i)'%(station,M)
#    else:
#        tt = 'STD by simulation vs CRB-STD\nfor %i stations of the IMS'%(llistIS)
#        
#    plt.title(tt)#,fontdict={'name':'Times New Roman'}) 
#    
#    ttexp = 'M = %i\nSNR = %i dB\nazimuth = %2.1f\nelevation = %2.1f\nvelocity = %2.1f m/s\nduration = %2.1f s' \
#                 %(M,results[0].SNR, results[0].az_deg, results[0].el_deg, 
#                   results[0].c_mps, results[0].dur_sec)
##    plt.text(0.7,0.05,ttexp)
#  
#plt.hold('False')

#======================================================================
##%%
#dirsavefigures = '/Users/maurice/etudes/stephenA/propal2/figures/'
#HorizontalSize = 5
#VerticalSize   = 3
#figSTDasSNR = plt.figure(num=2,figsize=
#               (HorizontalSize,VerticalSize), edgecolor='k', 
#                facecolor = [1,1,0.92])
#                
#plt.plot(listSNR0_dB, 20*log10(stdazCRB_deg[0,:]),'.-b',label='CRB')
#plt.hold('True')
#plt.plot(listSNR0_dB, 20*log10(stdazMC_deg[0,:]),'.-r',label='MC')
#plt.hold('False')
#plt.legend()
#plt.grid()
#plt.xlabel('SNR - dB')
#plt.ylabel('STD - dB')
#plt.show()
#if 0:
#    figSTDasSNR.savefig(dirsavefigures + \
#      'STDasSNRLOC%i%s.pdf'%(flag_LOC,station))

#%%
#plt.plot(listSNR0_dB,stdazCRB_deg[0,:]/stdazMC_deg[0,:],'.-b',label='CRB')
