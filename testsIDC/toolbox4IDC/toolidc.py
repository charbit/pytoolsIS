# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:44:30 2016

@author: maurice
"""


class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

from numpy import size, zeros, pi, cos, sin, int, intc, array, max
from numpy import ceil, exp, real, nan, std, log10, log2
from numpy import mean, conj, concatenate
from numpy import dot, transpose, diag, sqrt, random, ones

from scipy.linalg import svd
from scipy import fft, ifft , any, isreal



from numpy import append
from scipy.signal import lfilter, butter
from statsmodels.regression import yule_walker

#from scipy.integrate import quad
#from scipy.stats import norm
#from scipy.signal import hamming, hanning, resample, correlate
#from numpy.linalg import norm as norm2
#from numpy.linalg import matrix_rank, sqrtm, eigh, inv, pinv
#from numpy import linspace, logspace, sum, eye, kron
#from numpy import argmax, unravel_index, arange, histogram, diff
#from numpy import inf, angle, sort, trace, nansum

#from time import sleep
#from matplotlib import pyplot as plt

#=========================================================================
def synthetizer(station, SOI, LOC, SON, 
                          hugeNoise, failure, emergent):                             
    """
    #=====================================================================
    #
    # SYNOPSIS:
    #    synthetizer(station, SOI, LOC, SON, hugeNoise, failure, emergent)
    #
    #=====================================================================
    # Inputs:
    # station consists of M structures each of them with the 
    #        following items:
    #        .sensorname = sensor name, ex. H1
    #        .geolocs = structure which contains 5 items:
    #           latitude_deg,longitude_deg, X_km,Y_km, Z_km
    #     station: structure
    #        .sensorname = sensor name
    #        .data = geoloc structure which contains 5 items:
    #          latitude_deg,longitude_deg, X_km,Y_km, Z_km
    #     SOI = struct(name = 'soiinfo', soiflag_real = SOIflag_real,
    #             soidatabase = SOI_database,
    #             noiseflag_real = NOISEflag_real,
    #             noise_database = NOISE_database,
    #             totaltime_sec = totalTime_sec,
    #             Fs_Hz = SOIFs_Hz, 
    #             nb_events = SOInb_events,
    #             durationrange_sec = SOIdurationrange_sec,
    #             frequencywidth_Hz_Hz = SOIfrequencywidth_Hz_Hz,
    #             SNRrange_dB = SOISNRrange_dB,
    #             azimuthList_deg = SOIazimuthList_deg,
    #             elevationList_deg = SOIelevationList_deg,
    #             velocityList_mps = SOIvelocityList_mps,
    #             flagtimeoverlap = SOIflag_Timeoverlap,
    #             duration_at_least_sec = SOIduration_at_least_sec,
    #             flag_soiphasedistortion = flag_SOIphasedistortion)
    #
    #     emergent: structure
    #             .flag: 0 or 1
    #             .min_sec
    #             .durationpercent (50#)
    #     LOC       = struct(name='locinfo', flag = flag_LOC,
    #              std_azimuth_deg = LOC_std_azimuth_deg,
    #              std_elevation_deg = LOC_std_elevation_deg, 
    #              std_velocity_mps = LOC_std_velocity_mps)
    #     SON       = struct(name='soninfo', flag = flag_SON,
    #              SNR_dB = SON_SNR_dB, azimuth_deg = SON_azimuth_deg,
    #              elevation_deg = SON_elevation_deg, velocity_mps = SON_velocity_mps,
    #              frequencyband_Hz = SON_frequencyband_Hz)
    #     hugeNoise = struct(name = 'hugenoiseinfo', 
    #                   flag = flag_hugeNoise,
    #                   SNR_dB = hugeNoise_SNR_dB,
    #                   probability = hugeNoise_probability)
    #     failure = struct(name='failureinfo', flag = flag_failure,
    #                 probability = failure_probability)
    #
    #=====================================================================
    # Outputs: structure record
    #     .signals: array N by M
    #         N: duration in number of sampling period
    #         M: number of station
    #     .events: structure of length NB_EVENTS
    #       station: structure
    #        .sensorname = sensor name
    #        .data = geoloc structure M by 5
    #          latitude_deg,longitude_deg,
    #          elevation_km, X_km,Y_km
    #       SOI = struct(name = 'soiinfo', 
    #             soiflag_real = SOIflag_real,
    #             soidatabase = SOI_database,
    #             noiseflag_real = NOISEflag_real,
    #             noise_database = NOISE_database,
    #             totaltime_sec = totalTime_sec,
    #             Fs_Hz = SOIFs_Hz, 
    #             nb_events = SOInb_events,
    #             durationrange_sec = SOIdurationrange_sec,
    #             frequencywidth_Hz_Hz = SOIfrequencywidth_Hz_Hz,
    #             SNRrange_dB = SOISNRrange_dB,
    #             azimuthList_deg = SOIazimuthList_deg,
    #             elevationList_deg = SOIelevationList_deg,
    #             velocityList_mps = SOIvelocityList_mps,
    #             flagtimeoverlap = SOIflag_Timeoverlap,
    #             duration_at_least_sec = SOIduration_at_least_sec,
    #             flag_soiphasedistortion =  structure
    #                    .flagphasedistorsion
    #                    .direct
    #                    .reverse
    #      emergent: structure
    #             .flag: 0 or 1
    #             .min_sec
    #             .durationpercent (50#)
    #     LOC       = struct(name='locinfo', flag = flag_LOC,
    #              std_azimuth_deg = LOC_std_azimuth_deg,
    #              std_elevation_deg = LOC_std_elevation_deg, 
    #              std_velocity_mps = LOC_std_velocity_mps)
    #     SON       = struct(name='soninfo', flag = flag_SON,
    #              SNR_dB = SON_SNR_dB, azimuth_deg = SON_azimuth_deg,
    #              elevation_deg = SON_elevation_deg, velocity_mps = SON_velocity_mps,
    #              frequencyband_Hz = SON_frequencyband_Hz)
    #     hugeNoise = struct(name = 'hugenoiseinfo', 
    #                   flag = flag_hugeNoise,
    #                   SNR_dB = hugeNoise_SNR_dB,
    #                   probability = hugeNoise_probability)
    #     failure = struct(name='failureinfo', flag = flag_failure,
    #                 probability = failure_probability)
    """
    Fs_Hz                       = SOI.Fs_Hz;
    totalTime_sec               = SOI.totaltime_sec
    
    M                           = len(station);
    
    xsensors_m                  = zeros([M,3])
    for im in range(M):
        ism = station[im];
        xsensors_m[im,:] = \
           1000.0*array([ism.geolocs.X_km, ism.geolocs.Y_km, ism.geolocs.Z_km])
    
    nb_events                   = SOI.nb_events;

    flagrealsoi                 = SOI.soiflag_real;
    flagrealnoise               = SOI.noiseflag_real;
    flagLOC                     = LOC.flag;
    flagSON                     = SON.flag;
    flagemergent                = emergent.flag;
    flagfailure                 = failure.flag;
    flagHugeNoise               = hugeNoise.flag;
    flagTimeoverlap             = SOI.flagtimeoverlap
    flag_SOIphasedistortion     = SOI.flag_phasedistortion
    flag_SOIsuccessiveazimuth   = SOI.flag_successiveazimuth


    LOC_std_azimuth_deg         = LOC.std_azimuth_deg;
    LOC_std_elevation_deg       = LOC.std_elevation_deg;
    LOC_std_velocity_mps        = LOC.std_velocity_mps;
    
    SON_azimuth_deg             = SON.azimuth_deg;
    SON_elevation_deg           = SON.elevation_deg;
    SON_velocity_mps            = SON.velocity_mps;
    SON_SNR_dB                  = SON.SNR_dB;
    SONfrequencyband_Hz         = SON.frequencyband_Hz;
    
    min_for_emergent_sec        = emergent.min_sec; # typical 100 seconds
    emergentlength_percent      = emergent.durationpercent; # typical 25
    
    HugeNoiseprobability        = hugeNoise.probability;
    HugeNoiseSNR_dB             = hugeNoise.SNR_dB;
    
    failureprobability          = failure.probability;
    
    SOI_azimuthrange_deg        = SOI.azimuthrange_deg
    SOI_elevationrange_deg      = SOI.elevationrange_deg
    SOI_velocityrange_mps       = SOI.velocityrange_mps
    
    SOI_durationrange_sec       = SOI.durationrange_sec
    SOI_duration_at_least_sec   = SOI.duration_at_least_sec
    SOI_SNRrange_dB             = SOI.SNRrange_dB
    SOI_frequencywidth_Hz       = SOI.frequencywidth_Hz;
    SOI_margeazumith_deg        = SOI.margeazumith_deg 

    
# we have to use later the structure with geolocation values    
#    xsensor_m                   = zeros(sensornumber,3);
#    for im = 1:sensornumber
#        xsensor_m(im,:) = 1000*xsensors(im).data([4 5 3]);
#    end
    
    #=====================================================================
    # The LOC model is based on the randomness of the slowness vector
    # (from the work of Mack an Flinn). More specifically the slowness
    # vector is gaussianly distributed around a deterministic value
    # associated to the pure delays.
    #
    # The delays are evaluated as integers of the sampling period.
    #
    # The phase distorsion is based on a all-pass filter
    # with a delay of about 0.5 second (about 170 m)
    #
    #=====================================================================
    
    # Butterworth filter order to generate SOI from white noise
    SOI_filter_order   = 2;
    # for analysing the SOI
    orderLPC           = 10;
    # total duration
    T_pts              = int(totalTime_sec*Fs_Hz);
    meanxsensors_m     = array(mean(xsensors_m,0),ndmin=2);
    vectonesM          = ones([M,1])
    vectonesK          = ones([nb_events,1])
    xsensors_center_m  = xsensors_m - dot(vectonesM,meanxsensors_m);
    
    # length of the value lists for azimuth, elevation and velocity
   
    #=====================================================================
    # all-pass filter whose phase is almost linear in the band
    # (0-3.5) Hz. The pure delay is about 0.5 second
    # corresponding to about 170 m
    #
    Reverseallpass = array([ 1.0000,   -1.3913,    1.1386,   -0.6637,    
        0.3000,   -0.1062,    0.0291,   -0.0057,    0.0007]);
    Directallpass  = Reverseallpass[range(len(Reverseallpass)-1,0,-1)];
    
    #=====================================================================
    SOI_durationmin_sec = SOI_durationrange_sec[0];
    SOI_durationmax_sec = SOI_durationrange_sec[1];
    #=====================================================================
    SOI_SNRmin_dB     = SOI_SNRrange_dB[0];
    SOI_SNRmax_dB     = SOI_SNRrange_dB[1];
    
     
#    lengthazimuthrange   = len(SOI_azimuthrange_deg);
#    lengthelevationrange = len(SOI_elevationrange_deg);
#    lengthvelocityrange  = len(SOI_velocityrange_mps);
 
    diffazimrange = SOI_azimuthrange_deg[1]-SOI_azimuthrange_deg[0];
    diffelevrange = SOI_elevationrange_deg[1]-SOI_elevationrange_deg[0];
    diffvelorange = SOI_velocityrange_mps[1]-SOI_velocityrange_mps[0];
  
    #=====================================================================
    #====== LOC parameters
    #=====================================================================
    if flagLOC:
        Sigma_aec = diag([LOC_std_azimuth_deg*pi/180, 
            LOC_std_elevation_deg*pi/180,
            LOC_std_velocity_mps]);
    else:
        LOC_std_azimuth_deg      = nan;
        LOC_std_elevation_deg    = nan;
        LOC_std_velocity_mps     = nan;
        
    #=====================================================================
    #====== random draw for TOA, duration, frequency band
    #====== SNR, DOA, etc of each event
    #=====================================================================
    #==== frequency bands of PMCC
    # the width is about 10# of the mid-frequency
    # frequencybands_Hz = [
    #     0.05 0.3;
    #     0.3 0.5;
    #     0.5 0.8;
    #     0.8 1;
    #     1.1 1.7;
    #     1.8 2.3;
    #     2.3 2.8;
    #     2.8 3.1;
    #     3.2 3.6;
    #     3.6 4];
    #================
    # randomly selected bandwidth
    # Frequency bandwidths
    # SOI_Flims_Hz = sort(rand(nb_events,2)*diff(SOIfrequencywidth_Hz_Hz)+
    #     SOIfrequencywidth_Hz_Hz(1),2);
    # freqwidth = diff(SOI_Flims_Hz,[],2);
    # freqmiddle = mean(SOI_Flims_Hz,2);
    # rho = 0.25;
    # for ik=1:nb_events
    #     if freqwidth(ik)<0.1*freqmiddle(ik)
    #         SOI_Flims_Hz(ik,2) = (2+rho)*SOI_Flims_Hz(ik,1)/(2-rho);
    #     end
    # end
    # duration_at_least_sec = 1 ./ freqwidth;
    
    SOI_Flims_Hz = vectonesK*SOI_frequencywidth_Hz;
    
    # TOA, randomly selected TOA duration list in second
    SOI_TOA_sec = zeros(nb_events);
    SOI_dur_sec = zeros(nb_events);
    
    IG_totattimeMUL = 0.95;
    
    if not(flagTimeoverlap):
        meanlength_sec = (IG_totattimeMUL*totalTime_sec\
                           -SOI_durationmax_sec)/nb_events;
        SOI_TOA_sec[0] = meanlength_sec * random.rand();
        SOI_dur_sec[0] = (SOI_durationmax_sec-SOI_durationmin_sec)\
                       *random.rand()+SOI_durationmin_sec;
        SOI_dur_sec[0] = max([SOI_dur_sec[0], SOI_duration_at_least_sec]);
        nexteventbegin = SOI_TOA_sec[0]+SOI_dur_sec[0];
        for ik in range (1,nb_events):
            SOI_TOA_sec[ik] = meanlength_sec\
                     * random.rand()+nexteventbegin;
            SOI_dur_sec[ik] = (SOI_durationmax_sec-SOI_durationmin_sec)\
                     * random.rand()+SOI_durationmin_sec;
            SOI_dur_sec[ik] = max([SOI_dur_sec[ik], \
                       SOI_duration_at_least_sec]);
            nexteventbegin  = SOI_TOA_sec[ik]+SOI_dur_sec[ik];
            if nexteventbegin > totalTime_sec-SOI_durationmax_sec:
                print('too many events for the total duration');
                return []

    else:
        SOI_TOA_sec = IG_totattimeMUL*totalTime_sec\
                 * random.rand(nb_events);
        SOI_dur_sec = (SOI_durationmax_sec-SOI_durationmin_sec)\
                 *random.rand(nb_events)+SOI_durationmin_sec;
        for ik in range(nb_events):
            SOI_dur_sec[ik] = max([SOI_dur_sec[ik],\
                    SOI_duration_at_least_sec]);
    
    SOI_TOA_pts = intc(SOI_TOA_sec * Fs_Hz);
    SOI_dur_pts = intc(SOI_dur_sec * Fs_Hz);
    SOI_dur_pts = 2*intc(SOI_dur_pts/2.0);
    
    # randomly selected SNRs
    SOI_SNRs_dB = (SOI_SNRmax_dB-SOI_SNRmin_dB)\
              *random.rand(nb_events,M)+SOI_SNRmin_dB;
    
    # randomly selected azimuth with avoidance of near consecutive values
    # if possible
    SOI_azimuths_deg = zeros(nb_events);
#    SOI_azimuths_deg[0] = SOI_azimuthrange_deg[intc(random.rand()\
#    *lengthazimuthrange)];
#    for ik in range(1,nb_events):
#        az_auxm1 = SOI_azimuths_deg[ik-1];
#        az_aux   = SOI_azimuthrange_deg[intc(random.rand()\
#        *lengthazimuthrange)];
#        if not(lengthazimuthrange==1):
#            diffaz   = min([abs(az_aux-az_auxm1),\
#            abs(360-az_aux-az_auxm1)]);
#            while diffaz<10:
#                az_aux = SOI_azimuthrange_deg[intc(random.rand()\
#                *lengthazimuthrange)];
#                diffaz = min([abs(az_aux-az_auxm1), \
#                abs(360-az_aux-az_auxm1)]);
#        SOI_azimuths_deg[ik] = az_aux;
    
    SOI_elevations_deg  = zeros(nb_events);
    SOI_velocities_mps  = zeros(nb_events);
    
    SOI_azimuths_deg[0] = SOI_azimuthrange_deg[0]\
            +diffazimrange*random.rand()            
    SOI_elevations_deg[0] = SOI_elevationrange_deg[0]\
            +diffelevrange*random.rand()
    SOI_velocities_mps[0] = SOI_velocityrange_mps[0]\
            +diffvelorange*random.rand()

    
    for ik in range(1,nb_events):
#        SOI_elevations_deg[ik] = SOI_elevationrange_deg[intc(random.rand()\
#        *lengthelevationrange)];
#        SOI_velocities_mps[ik] = SOI_velocityrange_mps[intc(random.rand()\
#        *lengthvelocityrange)];
        
        SOI_azimuths_deg[ik] = SOI_azimuthrange_deg[0]\
                  +diffazimrange*random.rand()
        if (diffazimrange>SOI_margeazumith_deg & flag_SOIsuccessiveazimuth):
            while abs(SOI_azimuths_deg[ik]-SOI_azimuths_deg[ik-1])< SOI_margeazumith_deg:
                SOI_azimuths_deg[ik] = SOI_azimuthrange_deg[0]\
                 +diffazimrange*random.rand()
            
        SOI_elevations_deg[ik] = SOI_elevationrange_deg[0]\
                +diffelevrange*random.rand()
        SOI_velocities_mps[ik] = SOI_velocityrange_mps[0]\
                +diffvelorange*random.rand()


    SOI_slowness_spm    = zeros([3,nb_events]);
    # randomly selected station for huge noise and failure
    sensor_HugeNoise   = zeros(nb_events);
    sensor_failures    = zeros(nb_events);
    # Filter coefficients to built the SOI
    # Butterworth filter has been chosen
    SOI_filterCoeffs   = []
    for ik in range(nb_events):
        num, den = butter(SOI_filter_order,
                [SOI_Flims_Hz[ik,0]/Fs_Hz, SOI_Flims_Hz[ik,1]/Fs_Hz], 
                     btype='pass')
        SOI_filterCoeffs.append(struct(num=num,den=den))
#        if any(abs(roots([SOI_filterCoeffs{2,ik}]))>1):
#            ik, abs(roots([SOI_filterCoeffs{2,ik}]))
        
#%%    
    #=====================================================================
    # Generation of NB_EVENTS SOIs
    #       flagrealsoi=0:
    #       either as second order filtering of white noise
    #       flagrealsoi=1:
    #       or selected from an external real signal database
    #
    # for synthetic each event, the M signals are performed
    #    - with/without LOC
    #    - with/without emergent front edge
    #=====================================================================
    SOI_signalsM = list();
    overlong_mult = 5;
    for ik in range(nb_events):
        # aux_ik is ONE signal long enough 
        #  with zero-mean and STD = 1
        if flagrealsoi:
            aux_ik = readDATABASE(realSOI.database,overlong_mult\
                   *SOI_dur_pts(ik),Fs_Hz);
        else:
      # filtered whitenoise as SOI
            whitenoise = random.randn(overlong_mult\
                      *int(SOI_dur_pts[ik]),1);
            aux_ik = lfilter(SOI_filterCoeffs[ik].num, 
                SOI_filterCoeffs[ik].den, whitenoise);
        # centered and normalized
        aux_ik    = aux_ik - mean(aux_ik);
        aux_ik    = aux_ik/std(aux_ik);
        
        if flagLOC:
            # if LOC we use the function GENELWWLOCWITHOUTDELAY
            auxLOC, theta0_spm, Gamma2_epsilon, cf = (
            genelwwLOCwithoutdelay(aux_ik, 
                     orderLPC, 
                     Fs_Hz, 
                     SOI_azimuths_deg[ik], 
                     SOI_elevations_deg[ik], 
                     SOI_velocities_mps[ik], 
                     Sigma_aec, xsensors_m))
            LOC.Gamma2_epsilon  = Gamma2_epsilon
            LOC.slowness_spm    = theta0_spm;
            LOC.theoreticalCf   = cf
            
            SOI_signalsM.append(auxLOC[(overlong_mult-1)\
                      *SOI_dur_pts[ik]+range(int(SOI_dur_pts[ik])),:]);
        
        else:
            SOI_signal_ik = \
                aux_ik[int((overlong_mult-1)*SOI_dur_pts[ik])\
                +array(range(int(SOI_dur_pts[ik])))];
            LOC.Gamma2_epsilon  = nan
            LOC.slowness_spm    = nan;
            LOC.theoreticalCf   = nan
  
            SOI_signalsM.append(SOI_signal_ik  * ones([1,M]));
              
        # emergent 
        if flagemergent & (SOI_dur_sec[ik] > min_for_emergent_sec):
            halfnfrontedge_pts = int((SOI_dur_sec[ik]\
                       *emergentlength_percent/100.0)*Fs_Hz/2);
            lambda_emergemt = -halfnfrontedge_pts/4.0;
            frontedge = 1.0 / \
               (1.0+exp(array(range(-halfnfrontedge_pts,halfnfrontedge_pts))\
               /lambda_emergemt));

            for im in range(M):
                rangeindexN = range(2*halfnfrontedge_pts);
                SOI_signalsM[ik][rangeindexN,im] = \
                     frontedge * SOI_signalsM[ik][rangeindexN,im];
        
    #=====================================================================
    # phase distorsion on one randomly selected sensor
    if  flag_SOIphasedistortion:
        for ik in range(nb_events):
            mselect_distphase = int(M*random.rand());
            SOI_signalsM[ik][:,mselect_distphase] \
                  = lfilter(Directallpass, Reverseallpass, 
                    SOI_signalsM[ik][mselect_distphase,:]);
        # linear phase distorsion (not yet)
    else:
        Reverseallpass       = nan;
        Directallpass        = nan;
        mselect_distphase    = nan;
    
    #=====================================================================
    # compute noise levels on each sensor and each event
    SOI_sigma_noise  = zeros([nb_events,M]);
    for ik in range(nb_events):
        for im in range(M):
            SOI_sigma_noise[ik,im] = (10.0**(-SOI_SNRs_dB[ik,im]/20.0));
    #=====================================================================
    # Background noise is present in the all duration of signals.
    backgroundnoise = zeros([T_pts,M]);
    SOIonly         = zeros([T_pts,M]);
    
    if flagrealnoise:
        for im in range(M):
            backgroundnoise_aux = readDATABASE(realNOISE.database,T_pts,Fs_Hz);
            backgroundnoise_aux = backgroundnoise_aux - mean(backgroundnoise_aux);
            backgroundnoise_aux = backgroundnoise_aux / std(backgroundnoise_aux);
            backgroundnoise[:,im] = backgroundnoise_aux;
        
        backgroundnoise = mean(mean(SOI_sigma_noise))*backgroundnoise;
    else:
        backgroundnoise = mean(mean(SOI_sigma_noise))*random.randn(T_pts,M);
        
    # Below we add SOIs in accordance with the sensor location delays
    #=====================================================================
    observationswithnoise = backgroundnoise;
    #=====================================================================
    # computation delays
    SOI_delays_sec = zeros([M,nb_events]);
    SOI_delays_pts = zeros([M,nb_events]);
    for ik in range(nb_events):
        #==== azimuth from the North in the direct clockwise
        cosa   = cos(SOI_azimuths_deg[ik]*pi/180.0);
        sina   = sin(SOI_azimuths_deg[ik]*pi/180.0);
        cose   = cos(SOI_elevations_deg[ik]*pi/180.0);
        sine   = sin(SOI_elevations_deg[ik]*pi/180.0);
        SOI_slowness_spm[:,ik] \
               = array([-sina*cose,-cosa*cose, 
                        sine]/SOI_velocities_mps[ik]);
        SOI_delays_sec[:,ik]\
               = dot(xsensors_center_m,SOI_slowness_spm[:,ik]);
        SOI_delays_pts[:,ik] = SOI_delays_sec[:,ik] * Fs_Hz;
    # save on SOI structure
    SOI.delays_pts = SOI_delays_pts
    

    
    # here we add SOI to the background noise with the appropriate delays
    for ik in range(nb_events):
        # if huge noise, select one sensor with low probability
        if flagHugeNoise & (random.rand()<HugeNoiseprobability):
            sensor_HugeNoise[ik] = random.randint(M);
            SOI_sigma_noise[ik,sensor_HugeNoise[ik]]\
            = 10.0 ** (-HugeNoiseSNR_dB/20.0);
        else:
            sensor_HugeNoise[ik] = -1;
        # failure on one sensor
        if flagfailure & (random.rand()<failureprobability):
           sensor_failures[ik] = random.randint(M);
           SOI_signalsM[ik][:,sensor_failures[ik]]=0;      
        else:
           sensor_failures[ik] = -1;
        # apply delays
        for im in range(M):
            # we apply first the integer part of the delays
            delay_ik_im = SOI_delays_sec[im,ik]*Fs_Hz
            delayintegerpart = int(delay_ik_im);
            id1   = SOI_TOA_pts[ik]+delayintegerpart;
            id1 = id1 * (1-(id1<0))
            id2 = id1 + SOI_dur_pts[ik];
            if id2 > T_pts:
                print('too many events for the total duration');
                return []

            # then we apply decimal part by delay function
            delaydecimalpart = delay_ik_im-delayintegerpart;
            SOI_signalsM[ik][:,im] \
                    = delayedsignalF(SOI_signalsM[ik][:,im], 
                                     delaydecimalpart);
            observationswithnoise[range(id1,id2),im]\
                    = SOI_signalsM[ik][:,im];           
            
            SOIonly[range(id1,id2),im] = SOI_signalsM[ik][:,im];
            NOISEonly = SOI_sigma_noise[ik,im]*random.randn(SOI_dur_pts[ik]);
            
            observationswithnoise[range(id1,id2),im]\
                  = observationswithnoise[range(id1,id2),im]\
                    + NOISEonly;
             
    #=====================================================================
    # Signal of nuisance is added to the all duration
    #=====================================================================
    if flagSON:
        SON_slowness_spm  = zeros([3,nb_events]);
        for ik in range(nb_events):
            cosa   = cos(SON_azimuth_deg*pi/180);
            sina   = sin(SON_azimuth_deg*pi/180);
            cose   = cos(SON_elevation_deg*pi/180);
            sine   = sin(SON_elevation_deg*pi/180);
            #==== azimuth from the North in the direct clockwise
            SON_slowness_spm   = array([-sina*cose, -cosa*cose, sine])\
            /SON_velocity_mps;
        
        SON_delays_s        = dot(xsensors_center_m,SON_slowness_spm);
        SON_delays_pts      = SON_delays_s*Fs_Hz;
        SONnum, SONden      = butter(SOI_filter_order, \
        [SONfrequencyband_Hz[0]/Fs_Hz, \
         SONfrequencyband_Hz[1]/Fs_Hz],btype='pass')        
             
        whitenoise          = random.randn(size(observationswithnoise,0));
        SON_signal          = lfilter(SONnum,SONden, whitenoise);
        SON_signal          = SON_signal/std(SON_signal);
        SON_signals_wdelay  = zeros([size(observationswithnoise,0),M]);
        for im in range(M):
            SON_signals_wdelay[:,im] = delayedsignalF(SON_signal, \
            SON_delays_pts[im]);
                
        SON_sigma             = 10.0 ** (-SON_SNR_dB/20.0);
        SOI_withoutNoise      = SON_sigma * SON_signals_wdelay;
        observationswithnoise = observationswithnoise + SOI_withoutNoise;
      
#    #=====================================================================
#    #=====================================================================

    listevents = [];
#    
    for ik in range(nb_events):
        SOI = struct(failurenumerosensor = int(sensor_failures[ik]),
                hugeNoisenumerosensor = int(sensor_HugeNoise[ik]),
                azimuth_deg   = SOI_azimuths_deg[ik],
                elevation_deg = SOI_elevations_deg[ik],
                velocity_mps  = SOI_velocities_mps[ik],
                TOA_sec       = SOI_TOA_sec[ik],
                duration_sec  = SOI_dur_sec[ik],
                delayTOA_sec  = SOI_delays_sec[:,ik],#SOI_delays_sec[0,ik],
                SNR_dB        = -20*log10(SOI_sigma_noise[ik,:]));

        events = struct(
                SOI = SOI, LOC = LOC, SON = SON, hugeNoise = hugeNoise,
                failure = failure, emergent = emergent,
                
                phase_distorsion = struct(flagdistorsion=flag_SOIphasedistortion, \
                direct=Directallpass, reverse=Reverseallpass, \
                numsensor=mselect_distphase))
        
        listevents.append(events)

    return (observationswithnoise, listevents)

#=====================================================================
#=====================================================================
def genelwwLOCwithoutdelay(one_signal, orderLPC,
    Fs_Hz, az_deg, el_deg, velocity_mps, Sigma_aec, xsensors_m):
    """
    # Generate loss of coherence with noise
    # Synopsis:
    #   genelwwLOCwithoutdelay(one_signal, orderLPC,
    #        Fs_Hz, az_deg,el_deg, velocity_mps, Sigma_aec, xsensors_m)
    # Inputs
    #  ones_signal: one SOI
    #  orderLPC:
    #  Fs_Hz: sampling frequency in Hz
    #  az_deg, el_deg, velocity_mps:
    #         deterministic part of the DOA
    #
    #  Sigma_aec : 3 array of the varaince of the random part of the DOA
    #        Sigma_aec[0] = std of the azimuth in rd
    #        Sigma_aec[1] = std of the elevation in rd
    #        Sigma_aec[2] = std of the velocity in m/s
    #
    #  xsensor_m: M x 3
    #     3D locations of the M station sensors
    #
    #=======
    # Outputs:
    #  signal_withLOC:
    #          M-ary signal (with delays and LOC)
    #
    #  tau_s: TDOA on the C=M(M-1)/2 sensor pairs
    #          tau_s = (r_m-r_k)' x theta
    #      where theta is the wavenumber (in s/m)
    #
    #  signal_withoutLOC:
    #          M-ary signal (with delays but without LOC)
    #
    #  Gamma2_epsilon:
    #          positive matrix en s2/m2Ctrl+S (LOC)
    #=====================================================================
    """
    M          = size(xsensors_m,0);
    az_rd      = az_deg*pi/180;
    el_rd      = el_deg*pi/180;

    cosa       = cos(az_rd);
    sina       = sin(az_rd);
    cose       = cos(el_rd);
    sine       = sin(el_rd);        
    ##===== deterministic part
    theta0_spm = array([-sina*cose, -cosa*cosa, sine]/velocity_mps);
    ##===== random part
    Jacobian       = zeros([3,3])
    Jacobian[:,0]  = array([-cosa*cose, sina*sine, \
    sina*cose/velocity_mps]/velocity_mps);
    Jacobian[:,1]  = array([sina*cose, cosa*sine, \
    cosa*cose/velocity_mps]/velocity_mps);
    Jacobian[:,2]  = array([0, cose, \
    -sine/velocity_mps]/velocity_mps);
    Gamma2_epsilon = dot(transpose(Jacobian), \
    dot((Sigma_aec*Sigma_aec),Jacobian));
    ## AR analysis
    #%%
    N = len(one_signal);
    theta_lp, sigma_AR = yule_walker(one_signal,orderLPC);
    theta_AR = append(1,theta_lp)
    residue = lfilter(theta_AR,[1.0],one_signal)/sigma_AR;
    F_Hz = Fs_Hz*array(range(N))/N;
    F2_Hz2 = F_Hz ** 2;
    pi2 = pi*pi;
    #==== extract M random sequences from residue by permutation
    wwhite = zeros([N,M]);
    wwhite[:,0] = residue[:,0];
    for im in range(1,M):
        wwhite[:,im] = residue[random.permutation(N),0];
        
    #=== innovation generation
    cf = ones([M,M,N]);
    cp = 0;
    for im1 in range(M-1):
        for im2 in range(im1+1,M):
            cp = cp+1;
            sigma2_s2 = dot(dot((xsensors_m[im2,:]-xsensors_m[im1,:]), 
                Gamma2_epsilon),(xsensors_m[im2,:]-xsensors_m[im1,:]));
    # LOC WITHOUT delays
            cf[im1,im2,:] = exp(-2*pi2*F2_Hz2*sigma2_s2);
            cf[im2,im1,:] = conj(cf[im1,im2,:]);

    Wf = fft(wwhite);
    Xf = zeros([N,M])+ complex(0,1)*zeros([N,M])
    #=== GG = zeros([M,M,N]);
    for indf in range(N/2): # 1:N/2+1,
        GGaux = cf[:,:,indf];
        UG,GGauxsvd,VG = svd(GGaux);
        sqrtmGGaux = dot(dot(UG,diag(sqrt(GGauxsvd))),VG);
        Xf[indf,:] = dot(Wf[indf,:],sqrtmGGaux);

    Xf[range(N/2+1,N),:]   = conj(Xf[range(N-N/2-1,0,-1),:]);
    Xf[N/2,:]              = real(Xf[N/2,:])/2.0;
    Xf[0,:]                = real(Xf[0,:])/2.0;
    xt                     = ifft(Xf, axis=0);

    signal_out_aux         = real(xt);
    signal_out_aux         = signal_out_aux-ones([N,1])\
    *mean(signal_out_aux);
    #=== synthesis of signal
    signal_out_with             = signal_out_aux;
    signal_withLOC              = signal_out_aux;
    for im in range(M):
        xe = signal_out_aux[:,im];
        signal_out_with[:,im]   = lfilter([1.0], theta_AR, xe);
        signal_withLOC[:,im]    = signal_out_with[:,im]\
        / std(signal_out_with[:,im]);
    
    return signal_withLOC, theta0_spm, Gamma2_epsilon, cf


#==============================================================
#==============================================================
def delayedsignalF(x,t0_pts):
#==============================================================
    """
     Delay a signal with a non integer value
     (computation in frequency domain)
     
     Synopsis:
              y=delayedsignalF(x,t0_pts)
     
     Inputs: x vector of length N
             t0_pts is a REAL delay
             expressed wrt the sampling time Ts=1:
               t0_pts = 1 corresponds to one time dot
               t0_pts may be positive, negative, non integer
               t0_pts>0: shift to the right
               t0_pts<0: shift to the left
     Rk: the length of FFT is 2^(nextpow2(N)+1
    """
    #
    # M. Charbit, Jan. 2010
    #==============================================================
    N         = len(x)
    p         = ceil(log2(N))+1;
    Lfft      = int(2.0**p);
    Lffts2    = Lfft/2;
    fftx      = fft(x, Lfft);
    ind       = concatenate((range(Lffts2+1), 
            range(Lffts2+1-Lfft,0)),axis=0)
    fftdelay  = exp(-2j*pi*t0_pts*ind/Lfft);
    fftdelay[Lffts2] = real(fftdelay[Lffts2]);
    ifftdelay        = ifft(fftx*fftdelay);
    y                = ifftdelay[range(N)];
    if isreal(any(x)):
        y=real(y)
    return y
    #==============================================================