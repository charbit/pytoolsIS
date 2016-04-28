# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 07:35:28 2016

@author: maurice
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:28:09 2016

@author: maurice
#============================================
Compute the CRB as a functionn of the azimuth.
We consider that the full slowness vector is unknown and the 
two following scenarios:
   - without LOC
   - with LOC
#============================================

"""


class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
               
import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolutilities')

from geoloc import extractstationlocations

from toolIS import evalCRBwithgaussianLOC

from numpy import array,ones, exp, size, pi, arange, min, argmin
from numpy import random, mean, std, sum, log10
from numpy import zeros, dot, diag, cos, sin, sqrt, linspace, logspace
from matplotlib import pyplot as plt


listIS = ('I31',)

llist = len(listIS)

#===========================
Fs_Hz = 20.0;
SNR_dB = -10.0;
T_sec = 30.0;
aec = struct(e_deg = 70.0, c_mps = 340.0)
std_aecwithLOC = struct(a_deg = 5.0, e_deg = 3.0, c_mps = 13.0)
#===========================

results = []

for station in listIS:
    sensors = extractstationlocations(station, ReferenceEllipsoid=23)
    nbsensors = len(sensors)
    xsensors_m = zeros([nbsensors,3])
    if nbsensors == 0:
        print 'no sensor in %s'%station
    else:
        for im in range(nbsensors):
            evi   = sensors[im]
            xsensors_m[im,:]=array([evi.geolocs.X_km,
                    evi.geolocs.Y_km, 
                    evi.geolocs.Z_km])*1000.0;    
#    if ii==22:
#        xsensors_m=xsensors_m[range(0,nbsensors,2),:]
#        nbsensors=size(xsensors_m,0)
#
        M = size(xsensors_m,0);
    
        xsensors_m_centered = xsensors_m - ones([M,1])*mean(xsensors_m,0)
        normfactor = std(xsensors_m_centered,0) ; 
        meannormfactor = mean(normfactor)
        xsensors_m_norm = xsensors_m_centered / meannormfactor
    #%%
            
        sigma2noise = (10.0**(-SNR_dB/10.0));
        listazimuth = linspace(0,360,60)
        La = len(listazimuth)
        
        listapertureXfactor = linspace(0.3,1.5,8)
        Laperture = len(listapertureXfactor)
        
        sigmaa_deg = zeros([La,Laperture])
        sigmav_mps = zeros([La,Laperture])
        meanstdazwrtaz = zeros([Laperture,2])
        meanstdvwrtaz = zeros([Laperture,2])
            
        for LOCflag in (0,1):
            if LOCflag:        
                std_aec = std_aecwithLOC
            else:
                std_aec = struct(a_deg = 0.0, e_deg = 0.0, c_mps = 0.0)
            for iap in range(Laperture):
                xsensors_iap_m = xsensors_m_centered*listapertureXfactor[iap]
                for ia in range(La):
                    aec.a_deg = listazimuth[ia];
                    CRB, Jacobav_k, C0 = evalCRBwithgaussianLOC(xsensors_iap_m, 
                                sigma2noise, aec, std_aec,  T_sec , Fs_Hz)
                    sigmaa_deg[ia,iap] = sqrt(CRB.aec[0,0])*180/pi
                    sigmav_mps[ia,iap] = sqrt(CRB.av[1,1])
            
            meanstdazwrtaz[:,LOCflag] = mean(sigmaa_deg,axis=0)
            meanstdvwrtaz[:,LOCflag] = mean(sigmav_mps,axis=0)
        
            #%%
        #    HorizontalSize = 6
        #    VerticalSize   = 4
            
            argminsigma = argmin(sum(sigmaa_deg,axis=0))
            
            #figazimuth=plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
            #plt.subplot(1,2,1)
            #for iap in range(Laperture):
            #    rho = ((sigmaa_deg[:,iap]))
            #    rhoax0 = rho * cos(listazimuth*pi/180)
            #    rhoay0 = rho * sin(listazimuth*pi/180)
            #    if iap == argminsigma:
            #        plt.plot(rhoax0,rhoay0, label='radius x %i'%listaperture[iap])
            #    else:
            #        plt.plot(rhoax0,rhoay0,':', label='radius x %i'%listaperture[iap])
            #    plt.hold('True')
            #
            #plt.hold('False')
            ##plt.xlim([-2.0,2.0])
            ##plt.ylim([-2.0,2.0])
            #plt.xlabel('azimuth STD - degree',fontsize=8)
            #plt.ylabel('azimuth STD - degree',fontsize=8)
            #plt.grid()
            ##plt.legend(loc='best')
            #ax=plt.gca()
            #ax.yaxis.set_label_position('left')
            #plt.title('X %2.1f'%(listaperture[argminsigma]))
            #
            #plt.subplot(1,2,2)
            #plt.plot(xsensors_m_centered[:,0]/1000,xsensors_m_centered[:,1]/1000,'or')
            #plt.grid()
            #plt.xlabel('km',fontsize=8)
            #plt.ylabel('km',fontsize=8)
            #plt.title('station : %s, M = %i'%(station,M))
            
            #%%
        dirsavefigures = '/Users/maurice/etudes/stephenA/propal2/figures/'
        HorizontalSize = 5
        VerticalSize   = 3
        for LOCflag in (0,1):
            figmeanstdaz=plt.figure(num=2,figsize=
               (HorizontalSize,VerticalSize), edgecolor='k', 
                facecolor = [1,1,0.92]);
            plt.plot(listapertureXfactor,meanstdazwrtaz[:,LOCflag],'.-')
            plt.xlabel('multiplicative factor',fontsize=10)
            plt.ylabel('mean of STD - degree',fontsize=10)
            plt.grid()    
            if LOCflag:
                plt.title('with LOC')
                plt.show()
            else:
                plt.title('witouth LOC')      
                plt.show()
            figmeanstdaz.savefig(dirsavefigures + \
                 'CRBstdLOCasXfcactorLOC%i%s.pdf'%(LOCflag,station))
