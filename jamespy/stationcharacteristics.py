# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:06:19 2016

@author: maurice
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

from numpy import array, dot, size, ones, angle, std, mean, diff, sort, zeros
from numpy import histogram, log2, pi, cos, sin, sqrt
from numpy.linalg  import norm as norm2
from scipy.linalg import eigh, svd, inv, pinv

#==================================
def stationcharacteristics(xsensors_m, lisaz_deg):
        
    M = size(xsensors_m,0)
    combi = M*(M-1)/2;
    rangecombi=range(combi)
    regresscombi = array([ones(combi), rangecombi]);
    HtH_d = dot(regresscombi,regresscombi.transpose());
    HtHm1Ht = dot(inv(HtH_d),regresscombi);
    
    z = xsensors_m[:,0:2]
    
    HtH = M*dot(z.transpose(),z)    
    eigHtH = eigh(HtH)
    RR=eigHtH[0][1] / eigHtH[0][0]
    HtHproduct = eigHtH[0][1] * eigHtH[0][0]
    areaCov = 1.0 / HtHproduct
    
    distance=zeros(combi)
    orient=zeros(combi)
    
    cp=0;
    for im1 in range(M-1):
        for im2 in range(im1+1,M):
            distance[cp]=norm2(z[im1,:]-z[im2,:]);
            diffz = z[im1,:]+1j*z[im2,:]
            orient[cp]=angle(diffz[0]+1j*diffz[1]);
            cp=cp+1;
    dmin = min(distance)
    dmax = max(distance)
    
    dsort = sort(distance)
    diffsort_d = diff(dsort)
    dispersionratio_d = std(diffsort_d)/mean(diffsort_d)
    
    alpha_d = dot(HtHm1Ht,dsort)
    residu_d = dsort-dot(regresscombi.transpose(),alpha_d) 
    SSE_d = dot(residu_d,residu_d)
    SST_d = norm2(dsort-mean(dsort))**2;
    SSR_d=SST_d-SSE_d;
    R2_d = SSR_d/SST_d;
    
    osort = sort(orient)
    diffsort_o = diff(osort)
    dispersionratio_o = std(diffsort_o)/mean(diffsort_o)

    alpha_o = dot(HtHm1Ht,osort)
    residu_o = osort-dot(regresscombi.transpose(),alpha_o) 
    SSE_o = dot(residu_o,residu_o)
    SST_o = norm2(osort-mean(osort))**2;
    SSR_o=SST_o-SSE_o;
    R2_o = SSR_o/SST_o;

    histd=histogram(distance, bins=combi);
    histdnorm = histd[0]/float(combi);
    
    Hentropy_d = 0;
    for ic in range(combi):
        if not(histdnorm[ic]==0):
            Hentropy_d = Hentropy_d-histdnorm[ic]*log2(histdnorm[ic]);
    
    histo=histogram(orient, bins=combi);
    histonorm = histo[0]/float(combi);
    
    Hentropy_o = 0;
    for ic in range(combi):
        if not(histonorm[ic]==0):
            Hentropy_o = Hentropy_o-histonorm[ic]*log2(histonorm[ic]);
    
    HtH3d = dot(xsensors_m.transpose(),xsensors_m)
    invHtH3d = inv(HtH3d)
    
    La = len(lisaz_deg)
    STDCRBaz_deg = zeros(La);
    for ia in range(La):
        aec = struct(a_deg=lisaz_deg[ia],e_deg=45.0,c_mps=340);
        Jacobaec = jacobian3D(z, aec);
        invJacobaec = inv(Jacobaec);
        aux =  dot(dot(invJacobaec,invHtH3d),\
          invJacobaec.transpose());
          
        STDCRBaz_deg[ia] = sqrt(aux[0,0])*180.0/pi;
    
    
    return RR, areaCov, R2_d, R2_o, dmin, dmax, distance, \
          orient, Hentropy_d, Hentropy_o, dispersionratio_d, \
          dispersionratio_o, STDCRBaz_deg


#================================================          
def jacobian3D(xsensors_m, aec):

    a_rd = aec.a_deg*pi/180
    e_rd = aec.e_deg*pi/180
    cosa  = cos(a_rd);
    sina  = sin(a_rd);
    cose  = cos(e_rd);
    sine  = sin(e_rd);
    c_mps = aec.c_mps;
    
#    slowness_spm = aec2theta(a_rd,e_rd,c_mps)
#    delay_sec = dot(xsensors_m, slowness_spm);        
    
    # slowness_spm[0] = -sin(a_rd)*cos(e_rd)/c_mps
    # slowness_spm[1] = -cos(a_rd)*cos(e_rd)/c_mps
    # slowness_spm[2] = sin(e_rd)])/c_mps
    Jacobaec = array([
            [-cosa*cose/c_mps, sina*sine/c_mps, sina*cose/c_mps/c_mps],
            [sina*cose/c_mps, cosa*sine/c_mps, cosa*cose/c_mps/c_mps],
            [0.0, cose/c_mps, -sine/c_mps/c_mps]]);
    return Jacobaec
#===========================================
def aec2theta(a_rd,e_rd,c_mps):
#===========================================
    theta = array([-sin(a_rd)*cos(e_rd),
                   -cos(a_rd)*cos(e_rd),
                    sin(e_rd)])/c_mps
    return theta