# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:02:42 2016

@author: maurice
"""
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')

from toolIS import CRBonazimuthonlywithoutLOC, rotate2D, stationcharacteristics

from numpy import zeros, ones, cos, sin, max, min, sort, pi, sqrt
from numpy import size, angle, dot, mean, linspace, arange, log2
from numpy import random, array, setdiff1d
from numpy.linalg  import norm
from matplotlib import pyplot as plt
#from scipy.stats import linregress
from numpy.linalg import eigh, inv


#==================================
#def rotate2D(x,center,alpha_deg):
#    cosa = cos(alpha_deg*pi/180.0)
#    sina = sin(alpha_deg*pi/180.0)
#    Ralpha = array([[cosa, sina],[-sina, cosa]])
#    M=size(x,0)
#    y=zeros([M,2])
#    for im in range(M):
#        y[im,:]=center + dot(x[im,:]-center,Ralpha)
#    
#    return y
#
##==================================
#def stationcharacteristics(z):
#    M =size(z,0)
#    combi = M*(M-1)/2;
#    rangecombi=range(combi)
#    regresscombi = array([ones(combi), rangecombi]);
#    HtH_d = dot(regresscombi,regresscombi.transpose());
#    HtHm1Ht = dot(inv(HtH_d),regresscombi);
#    HtH = M*dot(z.transpose(),z)    
#    eigHtH = eigh(HtH)
#    RR=eigHtH[0][1] / eigHtH[0][0]
#    HtHproduct = eigHtH[0][1] * eigHtH[0][0]
#    areaCov = 1.0 / HtHproduct
#    
#    distance=zeros(combi)
#    orient=zeros(combi)
#    
#    cp=0;
#    for im1 in range(M-1):
#        for im2 in range(im1+1,M):
#            distance[cp]=norm(z[im1,:]-z[im2,:]);
#            diffz = z[im1,:]+1j*z[im2,:]
#            orient[cp]=angle(diffz[0]+1j*diffz[1]);
#            cp=cp+1;
#    dmin = min(distance)
#    dmax = max(distance)
#    dsort = sort(distance)
#    alpha_d = dot(HtHm1Ht,dsort)
#    residu_d = dsort-dot(regresscombi.transpose(),alpha_d) 
#    SSE_d = dot(residu_d,residu_d)
#    SST_d = norm(dsort-mean(dsort))**2;
#    SSR_d=SST_d-SSE_d;
#    R2_d = SSR_d/SST_d;
#    
#    osort = sort(orient)
#    alpha_o = dot(HtHm1Ht,osort)
#    residu_o = osort-dot(regresscombi.transpose(),alpha_o) 
#    SSE_o = dot(residu_o,residu_o)
#    SST_o = norm(osort-mean(osort))**2;
#    SSR_o=SST_o-SSE_o;
#    R2_o = SSR_o/SST_o;
#    
#    return RR, areaCov, R2_d, R2_o, dmin, dmax, distance, orient
#%%
#==================================


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

rangex0 = 1.7;
rangex1 = 0.75;
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

while ((R2_d < 0.9) | (R2_o < 0.9) | (RR>0.9*(rangex0/rangex1)) | (areaCov>10) |\
       (dmin<0.1*maxmax) | (dmax<0.7*maxmax))&(cr<10000):
    x[:,0] = rangex0*random.rand(M)-rangex0/2.0
    x[:,1] = rangex1*random.rand(M)-rangex1/2.0
    xc = x-dot(ones(M).reshape(M,1),mean(x,0).reshape(1,2))     
    
    # RR, areaCov, R2_d, R2_o, dmin, dmax, distance, \
#          orient, Hentropy_d, Hentropy_o, dispersionratio_d, dispersionratio_o
    RR, areaCov, R2_d, R2_o, dmin, dmax, distance, orient, Hd, Ho, \
            dispersionratio_d, dispersionratio_o = \
            stationcharacteristics(xc)
    cr=cr+1
    
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

STDCRBaz_deg = zeros(La)
for ia in range(La):
    aec = struct(a_deg=lisaz_deg[ia],e_deg=45.0,c_mps=340);
    CRBazonly = CRBonazimuthonlywithoutLOC(xsensors_m, 
                sigma2noise, aec, T_sec , Fs_Hz);
    STDCRBaz_deg[ia] = sqrt(CRBazonly)*180.0/pi

xrot = (STDCRBaz_deg) * cos(lisaz_deg*pi/180.0);
yrot = (STDCRBaz_deg) * sin(lisaz_deg*pi/180.0);

#%%
HorizontalSize = 8
VerticalSize   = 8

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
    
    plt.subplot(322)
    plt.plot(sort(distance))
    plt.hold('on')
    for im in range(M):
        plt.plot(sort(distance_f[im,:]))
        
    plt.hold('off')
    plt.grid('on')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('R2-d = %5.2f'%R2_d, fontsize=10)
        
    plt.subplot(324)
    plt.plot(sort(abs(orient))*180.0/pi,'.')
    plt.ylim([0,180])
    plt.grid('on')
    plt.xticks(fontsize=8)
    plt.yticks(arange(0,180.1,60),fontsize=8)
    plt.ylabel('degree',fontsize=8, horizontalalignment='right')
    plt.title('R2-o = %5.2f'%R2_o, fontsize=10)

    plt.subplot(223)
    plt.plot(xrot,yrot);
#    plt.hold(True)
    plt.axis('square');
    plt.xticks(arange(-4,4.1,1.0),fontsize=8)
    plt.yticks(arange(-4,4.1,1.0),fontsize=8)
    plt.xlim([-4.0,4.0])
    plt.ylim([-4.0,4.0])
    plt.grid('on')
    plt.xlabel('degree',fontsize=8)
    plt.title('area = %5.2f'% areaCov,fontsize=8)    
    plt.hold(False)   

    plt.subplot(326)
    plt.plot((RR_f),'.-')
    plt.grid('on')
    plt.xticks(fontsize=8)
    plt.hold ('on')
    plt.plot(array([0,M]),RR*array([1.0,1.0]),'--k')
    plt.hold ('off')

