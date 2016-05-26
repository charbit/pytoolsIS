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


from toolISmodules import CRBonazimuthonlywithoutLOC
from toolISmodules import evalCRBwithgaussianLOC, aec2theta
from stationcharacteristics import stationcharacteristics, jacobian3D


#%%
from numpy import zeros, ones, cos, sin, max, sort, pi, sqrt
from numpy import dot, mean, linspace, arange, log2, log10
from numpy import random, array, setdiff1d, histogram2d, nansum
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

rangex0 = 1.0;#1.7;
rangex1 = 1.0;#0.75;

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
dmax = 0;
dmin = 0;
maxmax = sqrt(rangex0**2+rangex1**2)

max_entropy = log2(M*(M-1)/2)

xsensors_m = zeros([M,3])
sigma2noise = 100.0;
T_sec=10.0;
Fs_Hz = 20.0;
La = 50;
lisaz_deg = linspace(0.0,360.0,La);
xsensors_m[:,2] = 30.0*random.rand(M);
stdaec = struct(a_deg=0.0,e_deg=0.0,c_mps=0.0)
STDCRBaz_deg = zeros(La)
T_pts = int(T_sec*Fs_Hz)


combi = M*(M-1)/2

RR_f1 = zeros(M)
areaCov_f1 = zeros(M)
R2_d_f1 = zeros(M)
R2_o_f1 = zeros(M)
dmin_f1 = zeros(M)
dmax_f1 = zeros(M)
distance_f1 = zeros([M,(M-1)*(M-2)/2]) 
orient_f1 = zeros([M,(M-1)*(M-2)/2])
Hd_f1 = zeros(M) 
Ho_f1 = zeros(M)
STDCRBazf1_deg = zeros([M,La]);

pairf2 = zeros([combi,2])
RR_f2 = zeros(combi)
areaCov_f2 = zeros(combi)
R2_d_f2 = zeros(combi)
R2_o_f2 = zeros(combi)
dmin_f2 = zeros(combi)
dmax_f2 = zeros(combi)
distance_f2 = zeros([combi, (M-2)*(M-3)/2]) 
orient_f2 = zeros([combi, (M-2)*(M-3)/2]) 
Hd_f2 = zeros(combi) 
Ho_f2 = zeros(combi)
STDCRBazf2_deg = zeros([combi,La]);

lrunsmax0 = 20000;

stopcondtxt0 = str('stopflag0= ( \
           (R2_d > 0.95) \
          & (R2_o > 0.95) \
          & (RR<1.2) \
          & (areaCov<0.03) \
          & (Hd_rel>0.8) \
          & (Ho_rel>0.8) \
          & (dmin>0.05*maxmax))')

#stopcondtxt0 = str('stopflag0=(areaCov<0.02)|(cr0>lrunsmax0-2)')
#stopcondtxt0 = str('stopflag0=(cr0>lrunsmax0-2)')

stopcondtxt1 = str('stopflag1=(max(areaCov_f1)<0.04)')

stopflag0 = False;
stopflag1 = False
cr0 = 0
H2d = zeros(lrunsmax0)
while not(stopflag1):
    while not(stopflag0| (cr0>lrunsmax0-2)):
        x[:,0] = rangex0*random.rand(M)-rangex0/2.0
        x[:,1] = rangex1*random.rand(M)-rangex1/2.0
        xc = x-dot(ones(M).reshape(M,1),mean(x,0).reshape(1,2))     
        xsensors_m[:,0:2] = xc;
    
        RR, areaCov, R2_d, R2_o, dmin, dmax, distance, orient, Hd, Ho, \
                dispersionratio_d, dispersionratio_o, STDCRBaz_deg  = \
                stationcharacteristics(xsensors_m,lisaz_deg)
        Hd_rel = Hd /max_entropy
        Ho_rel = Ho /max_entropy
        exec(stopcondtxt0)

        cr0=cr0+1
        
    if cr0>lrunsmax0-2:
        print "no solution"
        stopflag0 = True
        stopflag1 = True

    else:   
        print cr0
        for im in range(M):
            xcfail1 = xsensors_m[setdiff1d(range(M),(im,)),:]
            RR_f1[im], areaCov_f1[im], R2_d_f1[im], R2_o_f1[im],\
                dmin_f1[im], dmax_f1[im], distance_f1[im,:],\
                orient_f1[im,:], Hd_f1[im], Ho_f1[im],\
                dispersionratio_d, dispersionratio_o,\
                STDCRBazf1_deg[im,:] \
                = \
                stationcharacteristics(xcfail1, lisaz_deg)
        exec(stopcondtxt1)
        if not(stopflag1):
            stopflag0 = False     
            
#%%
cp=0
for im1 in range(M-1):
    for im2 in range(im1+1,M):
        pairf2[cp,:]=array([im1,im2])
        xcfail2 = xsensors_m[setdiff1d(range(M),(im1,im2)),:]
        
        RR_f2[cp], areaCov_f2[cp], R2_d_f2[cp], R2_o_f2[cp],\
            dmin_f2[cp], dmax_f2[cp], distance_f2[cp,:],\
            orient_f2[cp,:], Hd_f2[cp], Ho_f2[cp],\
            dispersionratio_d, dispersionratio_o,\
            STDCRBazf2_deg[cp,:]\
            = \
            stationcharacteristics(xcfail2, lisaz_deg)
        cp=cp+1
        
#%%
#========
if cr0<lrunsmax0-2:
    execfile('displayresults.py')
