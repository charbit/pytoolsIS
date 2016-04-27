# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:28:09 2016

@author: maurice
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
         
import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolutilities')

from geoloc import extractstationlocations

from toolIS import evalCRBwithgaussianLOC, CRBonazimuthonlywithoutLOC

from numpy import array,ones, exp, size, pi, arange
from numpy import zeros, dot, diag, cos, sin, sqrt, linspace
from matplotlib import pyplot as plt

#=================================================
Fs_Hz = 20.0;
SNR_dB = -10.0;
T_sec = 30.0;
sigma2noise = (10**(-SNR_dB/10.0));      
listazimuth = linspace(0,360,60)
aec = struct(e_deg = 70.0, c_mps = 340.0)
stdaec0 = struct(a_deg = 5.0, e_deg = 3.0, c_mps = 13.0)
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
#=================================================
La = len(listazimuth)
sigmaa_deg = zeros([La,2])
sigmav_mps = zeros([La,2])
sigmaaonly_deg = zeros([La,2])
for LOCflag in range(2):
    if LOCflag:        
        std_aec = stdaec0
    else:
        std_aec = struct(a_deg = 0.0, e_deg = 0.0, c_mps = 0.0)
    for ia in range(La):
        aec.a_deg = listazimuth[ia];
        CRB, Jacobav_k, C0known = evalCRBwithgaussianLOC(xsensors_m, 
                    sigma2noise, aec, std_aec,T_sec, Fs_Hz)
        CRBazonly = CRBonazimuthonlywithoutLOC(xsensors_m, 
                    sigma2noise, aec,T_sec , Fs_Hz)
        sigmaa_deg[ia,LOCflag] = sqrt(CRB.aec[0,0])*180/pi
        sigmaaonly_deg[ia,LOCflag] = sqrt(CRBazonly)*180/pi
        sigmav_mps[ia,LOCflag] = sqrt(CRB.av[1,1])
        
tt = '**************\nstation = %s\nSNR = %i dB\nazimuth = %2.1f\nelevation = %2.1f\nvelocity = %2.1f m/s\nduration = %2.1f s\nSTDaz = %2.3f' \
         %(station, SNR_dB, aec.a_deg, aec.e_deg, aec.c_mps, T_sec, sigmaa_deg[0,0])
print tt


#%%
HorizontalSize = 3
VerticalSize   = 3

figSTDasSNR=plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
rhoax0 = (sigmaa_deg[:,0]) * cos(listazimuth*pi/180)
rhoay0 = (sigmaa_deg[:,0]) * sin(listazimuth*pi/180)
plt.plot(rhoax0,rhoay0,'b', label='without LOC')
plt.hold('True')
rhoax1 = (sigmaa_deg[:,1]) * cos(listazimuth*pi/180)
rhoay1 = (sigmaa_deg[:,1]) * sin(listazimuth*pi/180)
plt.plot(rhoax1,rhoay1,'r', label='with LOC')
plt.hold('False')
plt.legend(loc='best',fontsize=8)
plt.axis('square')
#plt.xticks(arange(-0.3,0.31,0.2))
#plt.yticks(arange(-0.3,0.31,0.2))
plt.xlabel('azimuth STD - degree',fontsize=8)
plt.ylabel('azimuth STD - degree',fontsize=8)
ax=plt.gca()
ax.yaxis.set_label_position('right')
#plt.axis('square')
plt.grid()
if 0:
    dirsavefigures = '/Users/maurice/etudes/stephenA/propal2/figures/'
    figSTDasSNR.savefig(dirsavefigures + 'CRBonazimut.pdf')
#=======================
#figvitesse=plt.figure(num=2,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
#rhovx0 = (sigmav_mps[:,0]) * cos(listazimuth*pi/180)
#rhovy0 = (sigmav_mps[:,0]) * sin(listazimuth*pi/180)
#plt.plot(rhovx0,rhovy0,'b', label='without LOC')
#plt.hold('True')
#rhovx1 = (sigmav_mps[:,1]) * cos(listazimuth*pi/180)
#rhovy1 = (sigmav_mps[:,1]) * sin(listazimuth*pi/180)
#plt.plot(rhovx1,rhovy1,'r', label='with LOC')
#plt.hold('False')
#plt.legend(loc='best',fontsize=10)
#plt.axis('square')
#plt.grid()