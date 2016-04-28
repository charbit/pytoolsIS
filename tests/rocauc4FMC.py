# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:22:47 2016

@author: maurice
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:26:22 2016

@author: maurice

Synopsis:
This program performs the ROC curves for the
3 following functions of test:
   - Fstat
   - MCCM
   - consistence

"""
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)


         
import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolutilities')


from geoloc import extractstationlocations 
from toolIS import evalCRBwithgaussianLOC,maxfstat, UpsilonXi, geneFZ, geneFF
from toolIS import pvalunderH0, consistence, MCCM, synthetizer, rocauc

from numpy import array,ones, exp, size, pi, arange, sort
from numpy import zeros, dot, diag, cos, sin, sqrt, linspace
from numpy import random, std
from numpy import angle, arcsin, mod
from numpy.linalg import pinv
from numpy.linalg import norm as nm

from scipy.stats import f, norm

from matplotlib import pyplot as plt

#====================================
station = 'I31'
sensors = extractstationlocations(station, ReferenceEllipsoid=23)
nbsensors = len(sensors)
xsensors_m = zeros([nbsensors,3])
if nbsensors == 0:
    sys.exit('no sensor in %s'%station)
else:
    for im in range(nbsensors):
        evi   = sensors[im]
        xsensors_m[im,:] = array([evi.geolocs.X_km,
                  evi.geolocs.Y_km, 
                  evi.geolocs.Z_km])*1000.0;
    
    xsensors_m = xsensors_m - xsensors_m[0,:]
    M = size(xsensors_m,0);
flag_SON                    = 0;
flag_LOC                    = 0
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

LOC_std_azimuth_deg = 5.0;
LOC_std_elevation_deg = 3.0;
LOC_std_velocity_mps = 13.0;

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
SOISNRrange_dB              = -10.0*array([1.0, 1.0]);
SOI_margeazumith_deg         = 5;
SOIelevationrange_deg        = 70 * array([1, 1]);
SOIvelocityrange_mps         = 340 * array([1, 1]);
SOIduration_at_least_sec     = 0;
SOI_margeazumith_deg         = 10;
totalTime_s                  = 200.0;

Fs_Hz = 20
T_sec = SOIdurationrange_s[0] ; 
N = int(T_sec*Fs_Hz)
Lruns0 = 2;
Lruns1 = 100;


rangeel_deg = linspace(70,70,1)
rangevelo_mps = linspace(340,340,1)


pinvxsensors = pinv(xsensors_m[1:M,:])
#============== under H0 ===================
conssistence0  = zeros(Lruns0)
mccm0  = zeros(Lruns0)
fstat0 = zeros(Lruns0)
rangeaz0_deg = linspace(0,360, 100)
    
for ir0 in range(Lruns0):
    if (ir0-100*(ir0/100)) == 0: print ['0', ir0]
    signalsH0 = random.randn(N,M)
    aux0 = maxfstat(signalsH0, Fs_Hz, xsensors_m,
          rangeaz0_deg, rangeel_deg, rangevelo_mps)
    fstat0[ir0] = aux0[0];
 #   conssistence0[ir0], tkl = consistence(signalsH0)
    mccm0[ir0],tkl = MCCM(signalsH0)

#%%
#============== under H1 ===================
conssistence1  = zeros(Lruns1)
mccm1  = zeros(Lruns1)
fstat1 = zeros(Lruns1)
deltaazMLE = zeros(Lruns1)
deltaazMCCM = zeros(Lruns1)
for ir1 in range(Lruns1):
    if (ir1-100*(ir1/100)) == 0: print ['1', ir1]
    
    randazimuth1 = (random.randint(300)+30)
    SOIazimuthrange_deg = randazimuth1*ones(2)
    # to accelerate we assume that the maximal range is
    std_aec = struct(a_deg=0,e_deg=0,c_mps=0)
    aec = struct(a_deg=randazimuth1, e_deg=SOIelevationrange_deg[0],
                 c_mps = SOIvelocityrange_mps[0])
    sigma2noise = 10**(-SOISNRrange_dB[0]/10.0)           
    CRB = evalCRBwithgaussianLOC(xsensors_m, sigma2noise, aec, std_aec , 
                           SOIdurationrange_s[0], Fs_Hz)
    STDfromCRB_deg = sqrt(CRB[0].slowness[0,0])*180/pi
    
    rangeaz1_deg = linspace(randazimuth1-10.0,
                           randazimuth1+10.0, 50)
    
    #print STDfromCRB
    SOIazimuthrange1_deg = randazimuth1*array([1.0, 1.0]);
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

    signalsH1,listevents = synthetizer(sensors, SOI, LOC, 
              SON, hugeNoise, failure, emergent)
    evi1 = listevents[0];
    tstart_pts = int(evi1.SOI.TOA_sec*SOIFs_Hz);
    tdur_pts = int(evi1.SOI.duration_sec*SOIFs_Hz);
    tend_pts = tstart_pts+tdur_pts
    
    x1prime = signalsH1[tstart_pts:tend_pts,:]

    aux1 = maxfstat(x1prime, Fs_Hz, xsensors_m, 
          rangeaz1_deg, rangeel_deg, rangevelo_mps, visu=1)
    fstat1[ir1] = aux1[0];
    deltaazMLE[ir1]=abs(randazimuth1-aux1[1])   
    
#    conssistence1[ir1], tklc1 = consistence(x1prime)
    mccm1[ir1], tklm1 = MCCM(x1prime)
    
    hattheta = dot(pinvxsensors,tklm1[0:M-1]/20.0)    
    esta = angle(-hattheta[1]-1j*hattheta[0])*180.0/pi
    estc = 1.0 / nm(hattheta)
    este = arcsin(hattheta[2]*estc)*180.0/pi
    deltaazMCCM[ir1]=abs(randazimuth1-mod(esta,360))
    
print [std(deltaazMCCM),std(deltaazMLE)]
#%%

alpha_percent = 95
nbbins = 30
af, bf, CIalphaf, CIbetaf, eaucf, std_eauc_expf, std_eauc_bootf \
         = rocauc(fstat0,fstat1,alpha_percent,nbbins=nbbins)

ac, bc, CIalphac, CIbetac, eaucc, std_eauc_expc, std_eauc_bootc \
         = rocauc(conssistence0,conssistence1,alpha_percent,nbbins=nbbins)

am, bm, CIalpham, CIbetam, eaucm, std_eauc_expm, std_eauc_bootm \
         = rocauc(mccm0,mccm1,alpha_percent,nbbins=nbbins)
#%%
HorizontalSize = 4
VerticalSize   = 4

figROCs=plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), 
                edgecolor='k', facecolor = [1,1,0.92]);

plt.plot(af,bf, '.-',label='fstat')
plt.hold('True')
plt.plot(ac,bc, '.-',label='consistence')
plt.plot(am,bm, '.-',label='mccm')
plt.hold('False')
plt.grid()
plt.legend(loc='best')
tt = 'FSTAT : EAUC = %.2f (STD = %.2f)\nConsistence : EAUC = %.2f (STD = %.2f)\n MCCM : EAUC = %.2f (STD = %.2f)\n'\
   %(eaucf, std_eauc_expf, eaucc, std_eauc_expc, eaucm, std_eauc_expm)
plt.title(tt)

dirfigsave = '/Users/maurice/etudes/stephenA/propal2/figures'
hfile='%s/rocaucFMC%i.pdf' %(dirfigsave,flag_LOC)
#plt.figROCs(hfile,format='pdf')

#%%


#cosa=cos(evi1.SOI.azimuth_deg*pi/180.0)
#sina=sin(evi1.SOI.azimuth_deg*pi/180.0)
#cose=cos(evi1.SOI.elevation_deg*pi/180.0)
#sine=sin(evi1.SOI.elevation_deg*pi/180.0)
#c = evi1.SOI.velocity_mps
#thetaspm = array([-sina*cose,-cosa*cose, sine])/c;
#tau_sec = dot(xsensors_m,thetaspm)
#
#print [SOIazimuthrange_deg, esta]