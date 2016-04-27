# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 08:12:25 2016

@author: maurice
"""



class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
         
from numpy import size, zeros, complex, pi, cos, sin, intc, array, trace, nansum, max
from numpy import ceil, log2, exp, real, nan, std, log10, inf, nanmean, log, nanstd
from numpy import argmax, unravel_index, reshape, angle
from numpy import linspace, sum, mean, concatenate, conjugate
from numpy import dot, transpose, diag, sqrt, random, ones, eye, kron, append
from numpy import real, imag, bool, roots

from scipy.linalg import sqrtm, eigh, svd, inv
from scipy import fft, ifft , any, isreal, isnan, isinf
from numpy.linalg import matrix_rank, norm

from scipy.signal import lfilter, butter, cheby1, firwin
from scipy.signal import hamming, hanning

from matplotlib import pyplot as plt
from statsmodels.regression import yule_walker
from scipy.integrate import quad


import time

#function vmin = extract1direction(...
#    filteredsignalsUTk, filteredsignalsREFk, ...
#    HUTfk, HREFf, Vr)
def extract1direction(filteredsignalsUTk, filteredsignalsREFk, \
          HUTfk, HREFfk, Vr):
#=======================================================================
# Synopsis:
#    vmin = extract1direction(...
#     filteredsignalsUTk, filteredsignalsREFk, ...
#     HUTk, HREF, Vr)
# Inputs: 
#     - filteredsignalsUTk: 1 x N
#           signal of the selected SUT channel
#     - filteredsignalsREFk: 3 x N
#           signal of the 3 SREF channels 
#     - HUTk: 1 x N
#           frequency response of the selected SUT channel
#     - HREF: 3 x N
#           frequency response of the 3 SREF channels
#     - Vr: unitay vectors of the 3 SREF channels
#           each row is an unitary vector.
# Output
#     - vmin: 3 x 1
#           unitary vector in the direction
#           of the selected SUT channel solution 
#           of the least square problem.
# Used functions:
#      - roots, poly, norm from Matlab
#
#=======================================================================
    N       = size(filteredsignalsUTk,0);
    
    XUTfk   = array(zeros([N,3]),dtype=complex)
    XREFfk  = array(zeros([N,3]),dtype=complex)
    XUTfk   = fft(filteredsignalsUTk);

    for idir in range(3):
        XREFfk[:,idir]  = fft(filteredsignalsREFk[:,idir]);
    
    yurk    = (XUTfk / HUTfk) ; # deconvolution
    yREFf   = (XREFfk / HREFfk) ; # deconvolution
    # see document for notation
    Z       = dot(yREFf,inv(Vr));
    
    ZHZ     = dot(Z.transpose().conjugate(),Z);
    DD,UU   = eigh(ZHZ);
    w       = dot(yurk.conjugate(),Z);
      
    mu1     = DD[0];
    mu2     = DD[1];
    mu3     = DD[2];
    
    gamma1  = abs(dot(w,UU[:,0])) **2;
    gamma2  = abs(dot(w,UU[:,1])) **2;
    gamma3  = abs(dot(w,UU[:,2])) **2;
        
    
    Polylambda    = zeros(7);
    Polylambda[0] = 1;
    Polylambda[1] = 2*mu1 + 2*mu2 + 2*mu3;
    
    Polylambda[2] = mu1**2 + 4*mu1*mu2 + 4*mu1*mu3 + mu2**2 + \
        4*mu2*mu3 + mu3**2 - gamma1 - gamma2 - gamma3;
        
    Polylambda[3] = 2*mu1*(mu2**2) - 2*gamma2*mu1 - 2*gamma1*mu3 - \
        2*gamma3*mu1 - 2*gamma2*mu3 - 2*gamma3*mu2 - \
        2*gamma1*mu2 + 2*(mu1**2)*mu2 + 2*mu1*(mu3**2) + \
        2*(mu1**2)*mu3 + 2*mu2*(mu3**2) + 2*(mu2**2)*mu3 + 8*mu1*mu2*mu3;
        
    Polylambda[4]=(mu1**2)*(mu2**2) - gamma2*(mu1**2) - gamma1*(mu3**2) - \
        gamma3*(mu1**2) - gamma2*(mu3**2) - gamma3*(mu2**2) - \
        gamma1*(mu2**2) + (mu1**2)*(mu3**2) + (mu2**2)*(mu3**2) + \
        4*mu1*mu2*(mu3**2) + 4*mu1*(mu2**2)*mu3 + 4*(mu1**2)*mu2*mu3 - \
        4*gamma1*mu2*mu3 - 4*gamma2*mu1*mu3 - 4*gamma3*mu1*mu2;
        
    Polylambda[5]=2*(mu1**2)*(mu2**2)*mu3 + 2*(mu1**2)*mu2*(mu3**2) - \
        2*gamma3*(mu1**2)*mu2 - 2*gamma2*(mu1**2)*mu3 + \
        2*mu1*(mu2**2)*(mu3**2) - 2*gamma3*mu1*(mu2**2) - \
        2*gamma2*mu1*(mu3**2) - 2*gamma1*(mu2**2)*mu3 - \
        2*gamma1*mu2*(mu3**2);
    Polylambda[6]=(mu1**2)*(mu2**2)*(mu3**2) - gamma3*(mu1**2)*(mu2**2) - \
        gamma2*(mu1**2)*(mu3**2) - gamma1*(mu2**2)*(mu3**2);
        
    
    nbroots = 6;
    rootsP  = roots(Polylambda);
    Jmin    = inf;
    vmin    = nan;
    for ii in range(nbroots):
        invZ  = inv(ZHZ+rootsP[ii]*eye(3))
        v_ii  = dot(invZ,w);
        J_ii  = norm(yurk-dot(Z,v_ii.conjugate()));
        if J_ii<Jmin:
            Jmin = J_ii;
            vmin = real(v_ii);
    return vmin

#=====================================================
# function V = matrixtrihedron(a_deg,e_deg)
def matrixtrihedron(a_deg,e_deg):
#=====================================================
# synopsis:
#     V = matrixtrihadron(a_deg,e_deg)
# V is a 3 x 3 matrix, whose rows are unitary vectors
# V(k,:) = [cosa(k)*cose(k) sina(k)*cose(k) sine(k)]
#=====================================================
#
    K     = 3;
    a_rd  = a_deg*pi/180;
    e_rd  = e_deg*pi/180;
    cosa  = cos(a_rd);
    sina  = sin(a_rd);
    cose  = cos(e_rd);
    sine  = sin(e_rd);    
    V     = zeros([K,3]);
    for k in range(K):
        V[k,:] = array([cosa[k]*cose[k], sina[k]*cose[k], sine[k]]);
        
    return V
#=====================================================

#===========================================================
#
#function vmin = extract1directionSCP(...
#    filteredsignalsUTk, filteredsignalsREFk, ...
#    HUTfk, HREFf, Vr)
def extract1directionSCP(filteredsignalsUTk, filteredsignalsREFk,HUTfk, HREFf, Vr):
#=======================================================================
# Synopsis:
#    vmin = extract1direction(...
#     filteredsignalsUTk, filteredsignalsREFk, ...
#     HUTk, HREF, Vr)
# Inputs: 
#     - filteredsignalsUTk: 1 x N
#           signal of the selected SUT channel
#     - filteredsignalsREFk: 3 x N
#           signal of the 3 SREF channels 
#     - HUTk: 1 x N
#           frequency response of the selected SUT channel
#     - HREF: 3 x N
#           frequency response of the 3 SREF channels
#     - Vr: unitay vectors of the 3 SREF channels
#           each row is an unitary vector.
# Output
#     - vmin: 3 x 1
#           unitary vector in the direction
#           of the selected SUT channel solution 
#           of the least square problem.
# Used functions:
#      - roots, poly, norm from Matlab
#
#=======================================================================

    N = size(filteredsignalsUTk,0);
    
    nbshift = 10;
    wshift = N/nbshift;
    Lw = int(wshift*2.0);
    Suuf  = 0.0;
#    MSC   = zeros([3,3,Lw]);
    Srrf  = zeros([3,3,Lw]);
    Sruf  = zeros([3,Lw], dtype=complex);
    winhan = hanning(Lw);
    winhan = winhan / sqrt(sum(winhan **2));
    winhan3 = dot(winhan.reshape(Lw,1), ones([1,3]));
    for iw in range(nbshift-1):
        id1   = iw*wshift;
        id2   = id1+Lw;
        xut   = filteredsignalsUTk[id1:id2] * winhan;
        xrt   = filteredsignalsREFk[id1:id2,:] * winhan3;
        yuf   = fft(xut);# ./ HUTfk;
        yrf   = fft(xrt);# ./ HREFf;
        Suuf  = Suuf + abs(yuf) ** 2;
        for ifq in range(Lw):
            zz1 = yrf[ifq,:].reshape([1,3])
            zz2 = yrf[ifq,:].conjugate().reshape([3,1])
            Srrf[:,:,ifq] = Srrf[:,:,ifq]+ real(dot(zz2,zz1));
            Sruf[:,ifq]   = Sruf[:,ifq]+zz1*(yuf[ifq].conjugate());       
    
    # maxeig = zeros(Lw,1);
    # for ifq=1:Lw
    #     MSC(:,:,ifq)  = Srrf(:,:,ifq)\(Sruf(:,ifq)*Sruf(:,ifq)')/Suuf(:,ifq);
    #     maxeig(ifq)   = max(eig(MSC(:,:,ifq)));
    # end
    # see document for notation
    meanSrrf = mean(Srrf,axis=2)/Lw;
    meanSruf = mean(Sruf,axis=1)/Lw;
    Z        = meanSrrf * inv(Vr);

    ZHZ      = dot(Z.transpose().conjugate(),Z);
    DD,UU    = eigh(ZHZ);
    w        = dot(meanSruf.conjugate(),Z);
   
#      Z'*(meanSruf);
    mu1      = DD[0];
    mu2      = DD[1];
    mu3      = DD[2];
    
    gamma1   = abs(dot(w,UU[:,0])) **2;
    gamma2   = abs(dot(w,UU[:,1])) **2;
    gamma3   = abs(dot(w,UU[:,2])) **2;    
    
    Polylambda    = zeros(7);
    Polylambda[0] = 1;
    Polylambda[1] = 2*mu1 + 2*mu2 + 2*mu3;
    
    Polylambda[2] = mu1**2 + 4*mu1*mu2 + 4*mu1*mu3 + mu2**2 + \
        4*mu2*mu3 + mu3**2 - gamma1 - gamma2 - gamma3;
        
    Polylambda[3] = 2*mu1*(mu2**2) - 2*gamma2*mu1 - 2*gamma1*mu3 - \
        2*gamma3*mu1 - 2*gamma2*mu3 - 2*gamma3*mu2 - \
        2*gamma1*mu2 + 2*(mu1**2)*mu2 + 2*mu1*(mu3**2) + \
        2*(mu1**2)*mu3 + 2*mu2*(mu3**2) + 2*(mu2**2)*mu3 + 8*mu1*mu2*mu3;
        
    Polylambda[4]=(mu1**2)*(mu2**2) - gamma2*(mu1**2) - gamma1*(mu3**2) - \
        gamma3*(mu1**2) - gamma2*(mu3**2) - gamma3*(mu2**2) - \
        gamma1*(mu2**2) + (mu1**2)*(mu3**2) + (mu2**2)*(mu3**2) + \
        4*mu1*mu2*(mu3**2) + 4*mu1*(mu2**2)*mu3 + 4*(mu1**2)*mu2*mu3 - \
        4*gamma1*mu2*mu3 - 4*gamma2*mu1*mu3 - 4*gamma3*mu1*mu2;
        
    Polylambda[5]=2*(mu1**2)*(mu2**2)*mu3 + 2*(mu1**2)*mu2*(mu3**2) - \
        2*gamma3*(mu1**2)*mu2 - 2*gamma2*(mu1**2)*mu3 + \
        2*mu1*(mu2**2)*(mu3**2) - 2*gamma3*mu1*(mu2**2) - \
        2*gamma2*mu1*(mu3**2) - 2*gamma1*(mu2**2)*mu3 - \
        2*gamma1*mu2*(mu3**2);
    Polylambda[6]=(mu1**2)*(mu2**2)*(mu3**2) - gamma3*(mu1**2)*(mu2**2) - \
        gamma2*(mu1**2)*(mu3**2) - gamma1*(mu2**2)*(mu3**2);
        
    
    nbroots = 6;
    rootsP  = roots(Polylambda);
    Jmin    = inf;
    vmin    = nan;
    for ii in range(nbroots):
        invZ  = inv(ZHZ+rootsP[ii]*eye(3))
        v_ii  = dot(invZ,w);
        J_ii  = norm(meanSruf-dot(Z,v_ii.conjugate()));
        if J_ii<Jmin:
            Jmin = J_ii;
            vmin = real(v_ii);
    return vmin



#    ZHZ     = Z'*Z;
#    [UU,DD] = eig(ZHZ);
#    w       = Z'*(meanSruf);
#    
#    mu1     = DD(1,1);
#    mu2     = DD(2,2);
#    mu3     = DD(3,3);
#    gamma1  = abs(w'*UU(:,1))^2;
#    gamma2  = abs(w'*UU(:,2))^2;
#    gamma3  = abs(w'*UU(:,3))^2;
#    
#    Polylambda    = zeros(7,1);
#    Polylambda(1) = 1;
#    Polylambda(2) = 2*mu1 + 2*mu2 + 2*mu3;
#    Polylambda(3) = mu1^2 + 4*mu1*mu2 + 4*mu1*mu3 + mu2^2 + ...
#        4*mu2*mu3 + mu3^2 - gamma1 - gamma2 - gamma3;
#    Polylambda(4) = 2*mu1*mu2^2 - 2*gamma2*mu1 - 2*gamma1*mu3 - ...
#        2*gamma3*mu1 - 2*gamma2*mu3 - 2*gamma3*mu2 - ...
#        2*gamma1*mu2 + 2*mu1^2*mu2 + 2*mu1*mu3^2 + ...
#        2*mu1^2*mu3 + 2*mu2*mu3^2 + 2*mu2^2*mu3 + 8*mu1*mu2*mu3;
#    Polylambda(5)=mu1^2*mu2^2 - gamma2*mu1^2 - gamma1*mu3^2 - ...
#        gamma3*mu1^2 - gamma2*mu3^2 - gamma3*mu2^2 - ...
#        gamma1*mu2^2 + mu1^2*mu3^2 + mu2^2*mu3^2 + ...
#        4*mu1*mu2*mu3^2 + 4*mu1*mu2^2*mu3 + 4*mu1^2*mu2*mu3 - ...
#        4*gamma1*mu2*mu3 - 4*gamma2*mu1*mu3 - 4*gamma3*mu1*mu2;
#    Polylambda(6)=2*mu1^2*mu2^2*mu3 + 2*mu1^2*mu2*mu3^2 - ...
#        2*gamma3*mu1^2*mu2 - 2*gamma2*mu1^2*mu3 + ...
#        2*mu1*mu2^2*mu3^2 - 2*gamma3*mu1*mu2^2 - ...
#        2*gamma2*mu1*mu3^2 - 2*gamma1*mu2^2*mu3 - ...
#        2*gamma1*mu2*mu3^2;
#    Polylambda(7)=mu1^2*mu2^2*mu3^2 - gamma3*mu1^2*mu2^2 - ...
#        gamma2*mu1^2*mu3^2 - gamma1*mu2^2*mu3^2;
#    
#    nbroots = 6;
#    rootsP  = roots(Polylambda);
#    Jmin    = inf;
#    vmin    = NaN;
#    for ii=1:nbroots
#        v_ii  = (ZHZ+rootsP(ii)*eye(3))\w;
#        J_ii  = norm(meanSruf-Z*v_ii);
#        if J_ii<Jmin
#            Jmin = J_ii;
#            vmin = real(v_ii);
#        end
#    end
#    
#
