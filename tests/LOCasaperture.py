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
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')


from geoloc import extractstationlocations

from toolIS import evalCRBwithgaussianLOC, CRBonazimuthonlywithoutLOC

from numpy import array,ones, exp, size, pi, arange, min, argmin
from numpy import random, mean, std, sum, log10, sort
from numpy import zeros, dot, diag, cos, sin, sqrt, linspace, logspace
from matplotlib import pyplot as plt

#===========================
listIS = ('I37',)

Fs_Hz = 20.0;
SNR_dB = -10.0;
T_sec = 30.0;
aec = struct(e_deg = 70.0, c_mps = 340.0)
std_aecwithLOC = struct(a_deg = 1.0, e_deg = 1.0, c_mps = 10.0)
std_aecwithoutLOC = struct(a_deg = 0.0, e_deg = 0.0, c_mps = 0.0)
Laperture = 8;
listapertureXfactor = linspace(0.7,2.2,Laperture)
La = 10
listazimuth = linspace(0,360,La)

#===========================
llist = len(listIS)
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
        combi = M*(M-1)/2;
        distance_m = zeros(combi)
        cp=0;
        for im in range(combi-1):
            for imp in range(im+1,M):
                auxv=xsensors_m_centered[im,:]-xsensors_m_centered[imp,:]
                distance_m[cp]=sqrt(dot(auxv,auxv.reshape(3,1)))
                cp=cp+1
    #%%
            
        sigma2noise = (10.0**(-SNR_dB/10.0));
                
        sigmaa_deg = zeros([La,Laperture])
        sigmav_mps = zeros([La,Laperture])
        meanstdazwrtaz = zeros([Laperture,2])
        meanstdvwrtaz = zeros([Laperture,2])
            
        for LOCflag in (0,1):
            if LOCflag:        
                std_aec = std_aecwithLOC
            else:
                std_aec = std_aecwithoutLOC
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
        dirsavefigures = '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/propalSA/'
        HorizontalSize = 6
        VerticalSize   = 6
        figmeanstdaz=plt.figure(num='station %s'%station,figsize=
           (HorizontalSize,VerticalSize), edgecolor='k', 
            facecolor = [1,1,0.92]);
        plt.subplot(223)
        plt.plot(xsensors_m_centered[:,0]/1000.0,xsensors_m_centered[:,1]/1000.0,'o')
        plt.grid('on')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('km',fontsize=10)        
        plt.subplot(224)
        plt.plot(sort(distance_m/1000.0),'.-')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid('on')
        for LOCflag in (0,1):
            plt.subplot(2,2,LOCflag+1)
            plt.plot(listapertureXfactor,meanstdazwrtaz[:,LOCflag],'.-')
            plt.xlabel('multiplicative factor',fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid('on')    
            plt.show()
            if not(LOCflag):
                plt.title('Station %s'%station,fontsize=12)
                plt.ylabel('mean of STD - degree',fontsize=10)
#            figmeanstdaz.savefig(dirsavefigures + \
#                 'CRBstdLOCasXfcactorLOC%i%s.pdf'%(LOCflag,station))
