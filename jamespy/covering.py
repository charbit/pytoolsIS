# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:02:42 2016

@author: maurice
"""
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
         
mypaths = ('/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS', \
          '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities' )
Lmypaths = len(mypaths)

import sys
for ip in range(Lmypaths):
    if mypaths[ip] not in sys.path:
        sys.path.append(mypaths[ip])


from toolISmodules import CRBonazimuthonlywithoutLOC, stationcharacteristics
from toolISmodules import evalCRBwithgaussianLOC

#%%
from numpy import zeros, ones, cos, sin, max, sort, pi, sqrt
from numpy import dot, mean, linspace, arange, log2
from numpy import random, array, setdiff1d
from matplotlib import pyplot as plt
from numpy.linalg import inv

#%%
#====================================================
M=8;
x=zeros([M,2]);
combi = M*(M-1)/2;

#cp=0;
#A=zeros([combi,M])
#for im1 in range(combi-1):
#    for im2 in range(im1+1,M):
#        A[cp,im1]=1.0
#        A[cp,im2]=-1.0;
#        cp=cp+1

rangex0 = 1; #1.7;
rangex1 = 1;#0.75;
oneshot = 1

# RR controls the eigenvalue ratio, therefore the isotropy of the CRB
#     closer to 1, more isotropic 
# areaCov controls the eigenvalue product, therefore the accuracy of estimation
#     slower the areaCov, more accurate the estimation
# R2_d and R2_o control the uniformity of the distance and orientation
#     distribution. Closer to 1, more uniform the distributions
#     The way we use to draw induces a priori a good uniformity.
# it is also possible to have a constraint on dmin and dmax
# the problem (today) that we can not size these values.
# initialization
R2_d = 1.0
R2_o = 0
RR = 8
areaCov = 100;
cr = 0
dmax = 0;
dmin = 0;
maxmax = sqrt(rangex0**2+rangex1**2)

RR_f = zeros(M)
areaCov_f = zeros(M)
R2_d_f = zeros(M)
R2_o_f = zeros(M)
dmin_f = zeros(M)
dmax_f = zeros(M)
distance_f = zeros([M,(M-1)*(M-2)/2]) 
orient_f = zeros([M,(M-1)*(M-2)/2])
Hd_f = zeros(M) 
Ho_f = zeros(M)
lrunsmax = 20000;

while ((R2_d < 0.9) | (R2_o < 0.9) | (RR>1.5) | (areaCov>0.06) |\
       (dmin<0.1*maxmax))&(cr<lrunsmax):
    x[:,0] = rangex0*random.rand(M)-rangex0/2.0
    x[:,1] = rangex1*random.rand(M)-rangex1/2.0
    xc = x-dot(ones(M).reshape(M,1),mean(x,0).reshape(1,2))     
    
    # RR, areaCov, R2_d, R2_o, dmin, dmax, distance, \
#          orient, Hentropy_d, Hentropy_o, dispersionratio_d, dispersionratio_o
    RR, areaCov, R2_d, R2_o, dmin, dmax, distance, orient, Hd, Ho, \
            dispersionratio_d, dispersionratio_o = \
            stationcharacteristics(xc)
    cr=cr+1
if cr == lrunsmax:
    print('no solutions')
else:    
    for im in range(M):
        xcfail1 = xc[setdiff1d(range(M),(im,)),:]
        RR_f[im], areaCov_f[im], R2_d_f[im], R2_o_f[im],\
                dmin_f[im], dmax_f[im], distance_f[im,:],\
                orient_f[im,:], Hd_f[im], Ho_f[im],\
                dispersionratio_d, dispersionratio_o = \
                stationcharacteristics(xcfail1)
        
    max_entropy = log2(M*(M-1)/2)
    #========
    print cr
    print '****************************'
    print '\tR2_d = %3.2f, R2_o = %3.2f, RR = %3.2f, area = %5.3f\n\tdmin = %3.2f, dmax = %3.2f, Hd ratio = %3.2f, Ho ratio = %3.2f' \
                 %(R2_d, R2_o, RR, areaCov, dmin, dmax, Hd/max_entropy, Ho/max_entropy)
    print '****************************'
    xsensors_m = zeros([M,3])
    sigma2noise = 100.0;
    T_sec=10.0;
    Fs_Hz = 20.0;
    La = 50;
    lisaz_deg = linspace(0.0,360.0,La);
    xsensors_m[:,0:2] = 1000.0*xc;
    xsensors_m[:,2] = 30.0*random.rand(M);
    stdaec = struct(a_deg=0.0,e_deg=0.0,c_mps=0.0)
    
    #%%
    HtH=dot(xsensors_m.transpose(),xsensors_m);
    T_pts = int(T_sec*Fs_Hz)
    invHtHs2 = inv(HtH)*sigma2noise/(T_pts**2)
    
    STDCRBaz_deg = zeros(La)
    STDCRBazBIS_deg = zeros(La);
    for ia in range(La):
        aec = struct(a_deg=lisaz_deg[ia],e_deg=45.0,c_mps=340);
        CRB, Jacobaec, Jacobav, C0 = evalCRBwithgaussianLOC(xsensors_m,\
             sigma2noise, aec, stdaec, T_sec, Fs_Hz)
        CRBazonly = CRBonazimuthonlywithoutLOC(xsensors_m, 
                    sigma2noise, aec, T_sec , Fs_Hz);
        STDCRBaz_deg[ia] = sqrt(CRB.aec[0,0])*180.0/pi
        
        invJacobaec = inv(Jacobaec);
        aux =  dot(dot(invJacobaec,invHtHs2),\
                  invJacobaec.transpose());
        STDCRBazBIS_deg[ia] = sqrt(aux[0,0])*180.0/pi;
    
    xrot = (STDCRBaz_deg) * cos(lisaz_deg*pi/180.0);
    yrot = (STDCRBaz_deg) * sin(lisaz_deg*pi/180.0);
    xrotBIS = (STDCRBazBIS_deg) * cos(lisaz_deg*pi/180.0);
    yrotBIS = (STDCRBazBIS_deg) * sin(lisaz_deg*pi/180.0);
    
    
    HorizontalSize = 8
    VerticalSize   = 10
    
    figOptimGeometry=plt.figure(num='CRB',figsize=(HorizontalSize,VerticalSize), 
                           edgecolor='k', facecolor = [1,1,0.92]);
    
    mxu= 1.1*max([rangex0/2.0,rangex1/2.0])
    #====================
    if 1:
        plt.figure(num='CRB')
        plt.clf()
        plt.subplot(221)
        for im in range(M):
            plt.plot(x[im,0],x[im,1],'o',markersize=9)
            plt.hold('on')
            plt.text(0.85*x[im,0],0.85*x[im,1],'%i'%im)
            
        plt.plot([-rangex0/2.0,rangex0/2.0], [rangex1/2.0,rangex1/2.0],linewidth=2,color='k',linestyle='--')
        plt.plot([rangex0/2.0,rangex0/2.0], [rangex1/2.0,-rangex1/2.0],linewidth=2,color='k',linestyle='--')
        plt.plot([rangex0/2.0,-rangex0/2.0], [-rangex1/2.0,-rangex1/2.0],linewidth=2,color='k',linestyle='--')
        plt.plot([-rangex0/2.0,-rangex0/2.0], [-rangex1/2.0,rangex1/2.0],linewidth=2,color='k',linestyle='--')
    #    
        plt.hold('off')    
        plt.grid('on')
        plt.axis('square')
        plt.xlim(mxu*array([-1.0,1.0]))
        plt.ylim(mxu*array([-1.0,1.0]))
        plt.title('dmin = %5.2f, dmax = %5.2f'%(dmin,dmax), fontsize=10)
        plt.xticks(fontsize=8)
        plt.xlabel('km',fontsize=10)
        plt.yticks(fontsize=8)
        plt.ylabel('km',fontsize=10)
        
        plt.subplot(422)
        plt.plot(sort(distance))
        plt.hold('on')
        for im in range(M):
            plt.plot(sort(distance_f[im,:]),'--')
            
        plt.hold('off')
        plt.grid('on')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('distances with R2-d = %5.2f'%R2_d, fontsize=10)
            
        plt.subplot(424)
        plt.plot(sort(abs(orient))*180.0/pi)
        plt.hold('on')
        for im in range(M):
            plt.plot(sort(abs(orient_f[im,:]))*180.0/pi,'--')
            
        plt.hold('off')

        plt.ylim([0,180])
        plt.grid('on')
        plt.xticks(fontsize=8)
        plt.yticks(arange(0,180.1,60),fontsize=8)
        plt.ylabel('degree',fontsize=8, horizontalalignment='right')
        plt.title('orientations with R2-o = %5.2f'%R2_o, fontsize=10)
    
        plt.subplot(223)
        plt.plot(xrot,yrot);
        plt.hold(True)
        plt.plot(xrotBIS,yrotBIS);
        plt.hold('False')

        plt.axis('square');
        plt.xticks(arange(-4,4.1,1.0),fontsize=8)
        plt.yticks(arange(-4,4.1,1.0),fontsize=8)
        plt.xlim([-4.0,4.0])
        plt.ylim([-4.0,4.0])
        plt.grid('on')
        plt.xlabel('degree',fontsize=8)
        plt.title('CRB for the full station, area = %5.2f'% areaCov,fontsize=10)    
        plt.hold(False)   
    
        plt.subplot(4,2,6)
        plt.plot(RR_f,'.-')
        plt.grid('on')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.hold ('on')
        plt.plot(array([0,M-1]),RR*array([1.0,1.0]),'--k')
        plt.hold ('off')
        plt.title('isotropy',fontsize=10)

        plt.subplot(4,2,8)
        plt.plot(areaCov_f,'.-')
        plt.grid('on')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel('number of the failing sensor')
        plt.hold ('on')
        plt.plot(array([0,M-1]),areaCov*array([1.0,1.0]),'--k')
        plt.hold ('off')
        plt.title('accuracy',fontsize=10)
