# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:09:34 2016
@author: maurice
"""
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

from numpy import size, zeros, pi, cos, sin, int, intc, array, trace, nansum, max
from numpy import ceil, log2, exp, real, nan, std, log10, inf
from numpy import argmax, unravel_index, arange
from numpy import linspace,logspace, sum, mean, conj, concatenate
from numpy import dot, transpose, diag, sqrt, random, ones, eye, kron

from scipy.linalg import sqrtm, eigh, svd, inv, pinv
from scipy import fft, ifft , any, isreal, isnan
from numpy.linalg import matrix_rank

from scipy.signal import hamming, hanning, resample, correlate

from numpy import append
from scipy.signal import lfilter, butter
from statsmodels.regression import yule_walker
from scipy.integrate import quad
from scipy.stats import norm

from time import sleep
from matplotlib import pyplot as plt

#==========================================================================
#==========================================================================
def maxfstat(x,Fs_Hz, xsensor_m, range_azimuth_deg, range_elevation_deg, 
             range_velocity_mps, visu=0):

    """
    This function set is associated to the Fisher statistic 
    which is the GLRT of the detection of spatially coherent 
    signals in white noise.
    When the DOA is known the distribution under H0 (noise only) is
    Fisher distributed with N x N(M-1) dof where M is the number of sensors
    and N rhe size of the observations.
    When the DOA is unknown, the distribution under H0 is approximated
    via the distributions geneFF aor geneFZ

    The delays are real multiple of the sampling period
    
    synopsis:
    [maxF_time, argAmax] = fstat(x, ...
         Fs_Hz, xsensor_m, ...
         range_azimuth_deg, range_elevation_deg, range_velocity_mps)
    
     GLRT stat maximization wrt a,e,c
     Inputs:
          - x signal of size (N x M)
          - Fs_Hz : sampling frequency
          - xsensor_m array M x 3: sensor locations in meter
          - range_azimuth_deg: azimut in degree of the slowness 
          - from matplotlib import pyplot as plt
          - range_elevation_deg: elevation in degree of the slowness 
          - range_velocity_mps: velocity in m/s of the slowness       
         
    Outputs
           maxF_time : max F-stat wrt slowness
           azimuthmax: arg of the max
           elevationmax: arg of the max
           velocitymax: arg of the max
    """
    N        = size(x,0)
    M        = size(x,1)
    Laz      = size(range_azimuth_deg,0)
    Lel      = size(range_elevation_deg,0)
    Lve      = size(range_velocity_mps,0)
    F_time   = zeros([Laz,Lel,Lve])
#    F_timebis = zeros([Laz,Lel,Lve])
    #
    for ia in range(Laz):
         a_rd = range_azimuth_deg[ia]*pi/180;
         for ie in range(Lel):
             e_rd = range_elevation_deg[ie]*pi/180;
             for iv in range(Lve):
                 c = range_velocity_mps[iv]
                 theta_spm = array(
                      [-sin(a_rd)*cos(e_rd),
                       -cos(a_rd)*cos(e_rd), 
                        sin(e_rd)])/c;
                 tau_s = dot(xsensor_m,theta_spm);
                 tau_pts = tau_s * Fs_Hz;                
                 xtilde = zeros([N,M]);
                 for im in range(M):
                     xtilde[:,im] = delayedsignalF(x[:,im],-tau_pts[im])
                 RN = dot(transpose(xtilde),xtilde)/N
                 TN = trace(RN)
                 SNonM = sum(sum(RN))/M  
                 F_time[ia,ie,iv]=(M-1)*SNonM/(TN-SNonM)
    
    Fmax = max(F_time)
    indmax = array(zeros(3))
    indmax = unravel_index(F_time.argmax(),F_time.shape)
    azimuthmax = range_azimuth_deg[indmax[0]]
    elavationmax = range_elevation_deg[indmax[1]]
    velocitymax = range_velocity_mps[indmax[2]]
    
    # parabolic interpolation
    if (indmax[0]>3) & (indmax[0]<Laz-2):
        yykl = F_time[indmax[0]-1:indmax[0]+2, indmax[1], indmax[2]];
        xxkl = (range_azimuth_deg[indmax[0]-1],
                     range_azimuth_deg[indmax[0]],
                     range_azimuth_deg[indmax[0]+1])
#        deltaaz = range_azimuth_deg[1]-range_azimuth_deg[0]
#        invH = array([[0.0, 1.0, 0.0], \
#            [-0.5, 0.0, 0.5], \
#            [0.5, -1.0, 0.5]])
#        alphakl = dot(invH,yykl)
#        x_correction = -alphakl[1]/2.0/alphakl[2]
#        azimuthmax_parabolic = (azimuthmax + x_correction*deltaaz);
#        Fmax_parabolic = dot([1, x_correction, x_correction**2],alphakl)      
        azimuthmax_parabolic,Fmax_parabolic, alphakl = \
           parabolicinterpolation(xxkl,yykl)
        
    else:
        azimuthmax_parabolic=azimuthmax
        Fmax_parabolic=Fmax

    if visu:
        nptsparabole = 31;
        xpar = linspace(xxkl[0],xxkl[2],nptsparabole);
        
        parabole = zeros(nptsparabole)
        for ip in range(nptsparabole):
            parabole[ip] = \
            dot(array([1, xpar[ip], xpar[ip] ** 2]),alphakl);

        plt.plot(range_azimuth_deg, F_time[:,0,0],'.-')
        plt.hold('True')
        plt.plot(xpar,parabole,':k')
        plt.plot(azimuthmax_parabolic,Fmax_parabolic,'vy')
        plt.hold('False')
        plt.grid()
        plt.xlim(xmin=range_azimuth_deg[indmax[0]-2],
                 xmax=range_azimuth_deg[indmax[0]+2])
        plt.ylim(ymin=Fmax*0.9998, ymax=Fmax*1.0001)
        
        plt.show()
        
    return array([Fmax_parabolic, azimuthmax_parabolic, elavationmax, velocitymax])

#==========================================================================
def parabolicinterpolation(x,y):
    
    H = zeros([3,3])
    H[0,:] = [1,x[0],x[0]**2]
    H[1,:] = [1,x[1],x[1]**2]
    H[2,:] = [1,x[2],x[2]**2]
    alpha = dot(inv(H),y)
    xmax = -alpha[1]/2.0/alpha[2]
    ymax = dot([1,xmax,xmax**2],alpha)
    return xmax,ymax, alpha

#==========================================================================
#def maxfstatdiscretedelay(x,Fs_Hz, xsensor_m, range_azimuth_deg, range_elevation_deg, 
#             range_velocity_mps):
#
#    """
#    This function set is associated to the Fisher statistic 
#    which is the GLRT of the detection of spatially coherent 
#    signals in white noise.
#    When the DOA is known the distribution under H0 (noise only) is
#    Fisher distributed with N x N(M-1) dof where M is the number of sensors
#    and N rhe size of the observations.
#    When the DOA is unknown, the distribution under H0 is approximated
#    via the distributions geneFF aor geneFZ
#    
#    The delays are integer multiple fo the sampling period  
#
#    synopsis:
#    [maxF_time, argAmax] = maxfstatdiscretedelay(x, ...
#         Fs_Hz, xsensor_m, ...
#         range_azimuth_deg, range_elevation_deg, range_velocity_mps)
#    
#     GLRT stat maximization wrt a,e,c
#     Inputs:
#          - x signal of size (N x M)
#          - Fs_Hz : sampling frequency
#          - xsensor_m array M x 3: sensor locations in meter
#          - range_azimuth_deg: azimut in degree of the slowness 
#          - from matplotlib import pyplot as plt
#          - range_elevation_deg: elevation in degree of the slowness 
#          - range_velocity_mps: velocity in m/s of the slowness       
#         
#    Outputs
#           maxF_time : max F-stat wrt slowness
#           azimuthmax: arg of the max
#           elevationmax: arg of the max
#           velocitymax: arg of the max
#    """
#    N        = size(x,0)
#    M        = size(x,1)
#    Laz      = size(range_azimuth_deg,0)
#    Lel      = size(range_elevation_deg,0)
#    Lve      = size(range_velocity_mps)
#    F_time   = zeros([Laz,Lel,Lve])
#    for ia in range(0,Laz):
#         a_rd = range_azimuth_deg[ia]*pi/180;
#         for ie in range(0,Lel):
#             e_rd          = range_elevation_deg[ie]*pi/180;
#             for iv in range(0,Lve):
#                 c         = range_velocity_mps[iv]
#                 theta_spm = array([-sin(a_rd)*cos(e_rd),
#                                    -cos(a_rd)*cos(e_rd),sin(e_rd)])/c;
#                 tau_s     = dot(xsensor_m,theta_spm);
#                  
#                 tau_pts   = tau_s * Fs_Hz;
#             
#                 tau_posi  = -tau_pts+max(tau_pts);
#                 Ntau      = int(N - min(tau_pts) + max(tau_pts));
#                 xtilde    = zeros([Ntau,M]);
#                 for im in range(M):
#                    id1    = int(tau_posi[im]);
#                    id2    = id1+N;
#                    xtilde[range(id1,id2),im] = x[:,im];
#                    
#                 RN = dot(transpose(xtilde),xtilde)/Ntau
#                 TN = trace(RN);
#                 SN = sum(sum(RN))/M;                     
#                 F_time[ia,ie,iv]=(M-1)*SN/(TN-SN)
#                 
#    Fmax         = max(F_time)
#    indmax       = array(zeros(3))
#    indmax       = unravel_index(F_time.argmax(),F_time.shape)
#    azimuthmax   = range_azimuth_deg[indmax[0]]
#    elavationmax = range_elevation_deg[indmax[1]]
#    velocitymax  = range_velocity_mps[indmax[2]]
#    return array([Fmax, azimuthmax, elavationmax, velocitymax])

#==========================================================================
#==========================================================================
def pvalunderH0(F, N, xsensor_m, Fs_Hz, 
                range_azimuth_deg, range_elevation_deg, range_velocity_mps):
    """
    performs the p-value
    Inputs:
         F: observed values
         N: size of the observations
         xsensor_m: sensor locations in meter, M x 3
         Fs_Hz: sampling frequency in Hz
         range_azimuth_deg: array of azimuts in degree
         range_elevation_deg: array of elevations in degree 
         range_velocity_mps: array of velocities in m/s                
    """
    Lsamples = 10000;
    samples  = geneFF(Lsamples, N, xsensor_m, Fs_Hz, range_azimuth_deg, 
                  range_elevation_deg, range_velocity_mps)
    
    LF   = len(F)
    pval = zeros(LF)
    for i in range(LF):
        pval[i] = 1.0-float(sum(samples<F[i]))/Lsamples
    
    return pval
    
#==========================================================================
#==========================================================================
def threshunderH0(alpha, rangef, N, xsensor_m, Fs_Hz,
                  range_azimuth_deg, range_elevation_deg, range_velocity_mps):
    """
    perform the threshold to reach the false alamr alpha
    Inputs:
        alpha: observed values
        rangef:[f_begin, f_end]
        N: size of the observations
        xsensor_m: sensor locations in meter, M x 3
        Fs_Hz: sampling frequency in Hz
        range_azimuth_deg, range_elevation_deg, range_velocity_mps                
    """
                      
    Ll       = 500
    linef    = linspace(rangef[0],rangef[1],Ll)   
    Lsamples = 10000;
    thres    = 1
    maxF     = geneFF(Lsamples, N, xsensor_m, Fs_Hz, range_azimuth_deg, 
      range_elevation_deg, range_velocity_mps)
    for i in range(Ll):
        aa      = maxF<linef[i]
        bb      = float(sum(aa))/Lsamples
        if bb < 1-alpha:
            thres = linef[i]
            bbapprox = bb
    return thres, bbapprox

#============================
def geneFZ(Lsamples, N, xsensor_m, Fs_Hz, range_azimuth_deg, 
                  range_elevation_deg, range_velocity_mps):
    """
    generate Lsamples for the asymptotic distribution of F
    using the distribution of the 2*G multivariate gaussian
    where G = La*Le*Lc
    Inputs:
        Lsamples: number of samples
        N: size of the observations
        xsensor_m: sensor locations in meter, M x 3
        Fs_Hz: sampling frequency in Hz
        range_azimuth_deg: array of azimuts in degree
        range_elevation_deg: array of elevations in degree 
        range_velocity_mps: array of velocities in m/s                
    """

    M    = size(xsensor_m,0)
    M2   = M*M
    La   = len(range_azimuth_deg)
    Le   = len(range_elevation_deg)
    Lc   = len(range_velocity_mps)
    Q    = La*Le*Lc
    twoQ = 2*Q;
    
    #=========================================
    # Qdico_pts: delay dictionary
    Qdico_pts  = zeros([Q,M]);
    # Wdico_pts: delay difference dictionary
    Wdico_pts = zeros([Q,M,M])
    # Upsilon matrix (15), (16) and (17)
    Upsilon    = zeros([twoQ,twoQ]);
    cpQ        = -1
    for ia in range(La):
        az_rd = range_azimuth_deg[ia]*pi/180
        for ie in range(Le):
            el_rd = range_elevation_deg[ie]*pi/180;
            for ic in range(Lc):
                cpQ          = cpQ + 1;
                ce_mps       = range_velocity_mps[ic];
                theta        = aec2theta(az_rd,el_rd,ce_mps)
                Qdico_pts[cpQ,:] = Fs_Hz * dot(xsensor_m, theta)
    for iq in range(Q):
        for im1 in range(M):
            for im2 in range(M):
                Wdico_pts[iq,im1,im2]=int(Qdico_pts[iq,im1]-Qdico_pts[iq,im2]);
    
    for q1 in range(Q):
        for q2 in range(Q):
            # indic array is defined by the expression
            # \mathds{1}(\tau_{q1,m1}=\tau_{q2,m2});
            indicq1q2  = zeros([M,M]);
            for m1 in range(M):
                for m2 in range(M):
                    indicq1q2[m1,m2] = indicq1q2[m1,m2]+(Wdico_pts[q1,m1,m2]==Wdico_pts[q2,m1,m2])
            
            Sall                   = sum(indicq1q2);
            Sdiag                  = sum(diag(indicq1q2))
            Upsilon[2*q1,2*q2]     = 2.0*Sall/M2
            Upsilon[2*q1,2*q2+1]   = 2.0*(Sdiag - Sall/M)/(M-1)/M
            Upsilon[2*q1+1,2*q2]   = 2.0*(Sdiag - Sall/M)/(M-1)/M
            Upsilon[2*q1+1,2*q2+1] = 2.0*(Sdiag * (1-2./M) + Sall/M2)/(M-1)/(M-1)
            
    #=== generation
    #=== warning : we have to divide by sqrt(N)
    UpsilonsqrtN   = real(sqrtm(Upsilon))/sqrt(N);
    W              = random.randn(Lsamples,twoQ);
    Z              = ones([Lsamples,twoQ])+dot(W,UpsilonsqrtN)
    F1             = Z[:,range(0,twoQ,2)] / Z[:,range(1,twoQ,2)];
    maxF1          = max(F1,1);
    return maxF1    
    
    
#=============================================================
def geneFF(Lsamples, N, xsensor_m, Fs_Hz, range_azimuth_deg, 
                  range_elevation_deg, range_velocity_mps):
    """
    generate Lsamples for the asymptotic distribution of F
    using the disribution of the G multivariate gaussian
    where G = La*Le*Lc
    Inputs:
        Lsamples: number of samples
        N: size of the observations
        xsensor_m: sensor locations in meter, M x 3
        Fs_Hz: sampling frequency in Hz
        range_azimuth_deg: array of azimuts in degree
        range_elevation_deg: array of elevations in degree 
        range_velocity_mps: array of velocities in m/s                
   """
    M    = size(xsensor_m,0)
    M2   = M*M
    La   = len(range_azimuth_deg)
    Le   = len(range_elevation_deg)
    Lc   = len(range_velocity_mps)
    Q    = La*Le*Lc
    twoQ = 2*Q;
    
    #=========================================
    # Qdico_pts: delay dictionary
    Qdico_pts  = zeros([Q,M]);
    # Wdico_pts: delay difference dictionary
    Wdico_pts = zeros([Q,M,M])
    # Upsilon matrix (15), (16) and (17)
    Upsilon    = zeros([twoQ,twoQ]);
    cpQ        = -1
    for ia in range(La):
        az_rd = range_azimuth_deg[ia]*pi/180
        for ie in range(Le):
            el_rd = range_elevation_deg[ie]*pi/180;
            for ic in range(Lc):
                cpQ          = cpQ + 1;
                ce_mps       = range_velocity_mps[ic];
                theta        = aec2theta(az_rd,el_rd,ce_mps)
                Qdico_pts[cpQ,:] = Fs_Hz * dot(xsensor_m, theta)
    for iq in range(Q):
        for im1 in range(M):
            for im2 in range(M):
                Wdico_pts[iq,im1,im2]=int(Qdico_pts[iq,im1]-Qdico_pts[iq,im2]);
    
    for q1 in range(Q):
        for q2 in range(Q):
            # indic array is defined by the expression
            # \mathds{1}(\tau_{q1,m1}=\tau_{q2,m2});
            indicq1q2  = zeros([M,M]);
            for m1 in range(M):
                for m2 in range(M):
                    indicq1q2[m1,m2] = indicq1q2[m1,m2]+(Wdico_pts[q1,m1,m2]==Wdico_pts[q2,m1,m2])
            
            Sall                   = sum(indicq1q2);
            Sdiag                  = sum(diag(indicq1q2))
            Upsilon[2*q1,2*q2]     = 2.0*Sall/M2
            Upsilon[2*q1,2*q2+1]   = 2.0*(Sdiag - Sall/M)/(M-1)/M
            Upsilon[2*q1+1,2*q2]   = 2*(Sdiag - Sall/M)/(M-1)/M
            Upsilon[2*q1+1,2*q2+1] = 2.0*(Sdiag * (1-2./M) + Sall/M2)/(M-1)/(M-1)
    
    J   = kron(eye(Q),array([1, -1]));
    Xi  = dot(dot(J,Upsilon),transpose(J))
        
    #=== generation
    #=== warning : we have to divide by sqrt(N)
    XisqrtN        = real(sqrtm(Xi))/sqrt(N);
    W              = random.randn(Lsamples,Q);
    
    F2             = ones([Lsamples,Q])+dot(W,XisqrtN)
    maxF2          = max(F2,1)

    return maxF2
    
    
    
    #==========================================================================
def pvalunderH0IS(F, nu, N, xsensor_m, Fs_Hz, 
                range_azimuth_deg, range_elevation_deg, range_velocity_mps):
    """
    performs the p-value par Importance Sampling
    Inputs:
         F: observed values
         N: size of the observations
         xsensor_m: sensor locations in meter, M x 3
         Fs_Hz: sampling frequency in Hz
         range_azimuth_deg: array of azimuts in degree
         range_elevation_deg: array of elevations in degree 
         range_velocity_mps: array of velocities in m/s                
    """
    La   = len(range_azimuth_deg)
    Le   = len(range_elevation_deg)
    Lc   = len(range_velocity_mps)
    Q    = La*Le*Lc
    
    Lsamples = 10000;
    
    Upsilon, Xi = UpsilonXi(xsensor_m, Fs_Hz, range_azimuth_deg, 
                  range_elevation_deg, range_velocity_mps)

    rankXi   = matrix_rank(Xi);
    muiid    = random.randn(Lsamples,rankXi);
    
    D,U = eigh(Xi)
    invD = zeros(Q);
    rangeinvD = range(Q-rankXi,Q)
    invD[rangeinvD] = 1.0 / D[rangeinvD]
    invD = diag(invD)
    Ureduc = U[:,rangeinvD]   
 
    pinvXi = dot(dot(transpose(Ureduc), invD),(Ureduc))
#    loga = log(pareto_a)
#    logm = pareto_a*log(pareto_m)
    w         = zeros(Lsamples)
    integrand = zeros(Lsamples)
    for i in range(Lsamples):
        vcur = muiid[i,:]-1;
        tcur = muiid[i,:];
        w1   = -dot(dot(transpose(vcur),pinvXi),vcur)/2;
        q1   = -rankXi*dot(transpose(tcur),tcur)/2;
        w[i] = exp(N*(w1-q1));
        integrand[i] = max(vcur);
    sumw = nansum(w)
    wnorm = w/sumw
    LF = len(F)
    pval = zeros(LF)
    aux = zeros(LF)
    for i in range(LF):
        lfi      = F[i]
        cdfF     = dot(wnorm,(integrand<lfi))
        aux[i]   = nansum(cdfF)
        pval[i]  = 1. - aux[i]
        
    return pval, integrand
 
#==========================================================   
def pareto_p(N,Q,pareto_a,pareto_m): # nu equals number of degrees of freedom
    """
    
    """
    t = zeros([N,Q])
    for i in range(N):
#        x = random.randn(1)
#        y = 2.0*random.gamma(0.5*nu, 2.0)
        t[i,:] = (random.pareto(pareto_a,Q)+1)*pareto_m
    return t


#============================================================
#=============================================================
def UpsilonXi(xsensor_m, Fs_Hz, range_azimuth_deg, 
                  range_elevation_deg, range_velocity_mps):
    """
    generate Xi and Upsilon matrix 
    using the disribution of the G multivariate gaussian
    where G = La*Le*Lc
    Inputs:
        Lsamples: number of samples
        N: size of the observations
        xsensor_m: sensor locations in meter, M x 3
        Fs_Hz: sampling frequency in Hz
        range_azimuth_deg: array of azimuts in degree
        range_elevation_deg: array of elevations in degree 
        range_velocity_mps: array of velocities in m/s                
   """
    M    = size(xsensor_m,0)
    M2   = M*M
    La   = len(range_azimuth_deg)
    Le   = len(range_elevation_deg)
    Lc   = len(range_velocity_mps)
    Q    = La*Le*Lc
    twoQ = 2*Q;
    
    #=========================================
    # Qdico_pts: delay dictionary
    Qdico_pts  = zeros([Q,M]);
    # Wdico_pts: delay difference dictionary
    Wdico_pts = zeros([Q,M,M])
    # Upsilon matrix (15), (16) and (17)
    Upsilon    = zeros([twoQ,twoQ]);
    cpQ        = -1
    for ia in range(La):
        az_rd = range_azimuth_deg[ia]*pi/180
        for ie in range(Le):
            el_rd = range_elevation_deg[ie]*pi/180;
            for ic in range(Lc):
                cpQ          = cpQ + 1;
                ce_mps       = range_velocity_mps[ic];
                theta        = aec2theta(az_rd,el_rd,ce_mps)
                Qdico_pts[cpQ,:] = Fs_Hz * dot(xsensor_m, theta)
    for iq in range(Q):
        for im1 in range(M):
            for im2 in range(M):
                Wdico_pts[iq,im1,im2]=int(Qdico_pts[iq,im1]-Qdico_pts[iq,im2]);
    
    for q1 in range(Q):
        for q2 in range(Q):
            # indic array is defined by the expression
            # \mathds{1}(\tau_{q1,m1}=\tau_{q2,m2});
            indicq1q2  = zeros([M,M]);
            for m1 in range(M):
                for m2 in range(M):
                    indicq1q2[m1,m2] = indicq1q2[m1,m2]+(Wdico_pts[q1,m1,m2]==Wdico_pts[q2,m1,m2])
            
            Sall                   = sum(indicq1q2);
            Sdiag                  = sum(diag(indicq1q2))
            Upsilon[2*q1,2*q2]     = 2.0*Sall/M2
            Upsilon[2*q1,2*q2+1]   = 2.0*(Sdiag - Sall/M)/(M-1)/M
            Upsilon[2*q1+1,2*q2]   = 2.0*(Sdiag - Sall/M)/(M-1)/M
            Upsilon[2*q1+1,2*q2+1] = 2.0*(Sdiag * (1-2./M) + Sall/M2)/(M-1)/(M-1)
    
    J   = kron(eye(Q),array([1, -1]));
    Xi  = dot(dot(J,Upsilon),transpose(J))
    return Upsilon, Xi


#===========================================
def aec2theta(a_rd,e_rd,c_mps):
#===========================================
    theta = array([-sin(a_rd)*cos(e_rd),
                   -cos(a_rd)*cos(e_rd),
                    sin(e_rd)])/c_mps
    return theta
#===========================================
def delayedsignalF(x,t0_pts):
#===========================================
    """
     delay a signal with a non integer value
     (computation in frequency domain)
     
     synopsis:
     y=delayedsignalF(x,t0_pts)
     
     inputs: x vector of length N
     the delay t0_pts is a REAL delay
     expressed wrt the sampling time Ts=1:
       t0_pts = 1 corresponds to one time dot
       t0_pts may be positive, negative, non integer
       t0_pts>0: shift to the right
       t0_pts<0: shift to the left
     Rk: the length of FFT is 2^(nextpow2(N)+1
     """
#
# M. Charbit, Jan. 2010
#==============================================
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
#===============================================
def generateIS(signal0, Fs_Hz, xsensor_m, azimuth_deg, elevation_deg, 
               velocity_mps):
    """
    Generate a IS signal on M sensors
    synopsis generateIS(signal0, Fs_Hz, xsensor_m, azimuth_deg, 
             elevation_deg, velocity_mps)
    Inputs:
       - signal0: one signal-source (array Nx1)
       - Fs_Hz: sampling frequency in Hz
       - xsensors (M x 3) array where M is the sensor number
       - azimuth_deg, elevation_deg, velocity_mps
    Output:
       - N x M array
    """
    M    = size(xsensor_m,0);
    N    = size(signal0,0);
    a_rd = azimuth_deg*pi/180
    e_rd = elevation_deg*pi/180
    c    = velocity_mps
    theta_spm = array([-sin(a_rd)*cos(e_rd),
                       -cos(a_rd)*cos(e_rd),
                        sin(e_rd)])/c;
    tau_s     = dot(xsensor_m, theta_spm);
    tau_pts   = tau_s * Fs_Hz;
    x    = zeros([N,M]);
    for im in range(M):
        x[:,im] = delayedsignalF(signal0,tau_pts[im])
    return x
    
#===============================================
def generateISsto(N,Fs_Hz, xsensor_m, azimuth_deg, 
                  elevation_deg,velocity_mps,
                  sigma_azimuth_deg, sigma_elevation_deg, 
                  sigma_velocity_mps, nbRays=50):
    """
    Generate IS signals on M sensors
    The slowness vector is drawn independently on each sensor
    The goal is to simulte the LOC by addition
    
    Synopsis generateISsto(signal0, Fs_Hz, xsensor_m, azimuth_deg, 
             elevation_deg, velocity_mps)
    Inputs:
       - signal0: signal-source (array Nx1)
       - Fs_Hz: sampling frequency in Hz
       - xsensors (M x 3) array where M is the sensor number
       - azimuth_deg, elevation_deg, velocity_mps
    Output:
       - N x M array
    """
    #Fs_Hz = float(Fs_Hz)
    M     = size(xsensor_m,0);
    a_rd0 = azimuth_deg*pi/180
    e_rd0 = elevation_deg*pi/180
    c0    = velocity_mps
    x     = zeros([N,M]);
    tau_pts = zeros([M,nbRays])
    for ir in range(nbRays):
        # for each infrasound ray we draw a new trajectory
        signal0 = random.randn(N)
        a_rd = a_rd0+sigma_azimuth_deg*pi*random.randn()/180
        e_rd = e_rd0+sigma_elevation_deg*pi*random.randn()/180
        c    = c0+sigma_velocity_mps*random.randn()
        theta_spm = array([-sin(a_rd)*cos(e_rd),
                           -cos(a_rd)*cos(e_rd),
                            sin(e_rd)])/c;

        xir = zeros([N,M])
        for im in range(M):
            tau_s = dot(xsensor_m[im], theta_spm);
            tau_pts[im,ir]   = tau_s * Fs_Hz;
            xir[:,im] =  delayedsignalF(signal0,-tau_pts[im,ir])
            
        x = x + xir
    
    x = x/nbRays;
    return x, tau_pts
    
#==========================================================================
def estimSCP(x1,x2,Lfft,overlapFFT, Fs_Hz, smoothwindow):
    """
     Perform the spectral components of the two signals x1 et x2.

     The code uses the Welch's approach. The signal is shared into DFT
     windows of which the length is Lfft, with the
     overlap rate of OVERLAPFFT. Then the specral components is averaged
     on some DFT blocks. Therefore each spectral block co22esponds
     to a time period reported in TIME_SEC.

     Inputs:
        - x1: signal observed on the SUT (T x 1)
        - x2: signal observed on the SREF (T x 1)
        - Lfft: length og the FFTs
        - overlapFFT: between 0 and 1, overlap on the FFT block
        - Fs_Hz: sampling frequency in Hz
        - smoothwindow: character word as 'rect', 'hanning', 
                hamming, ...
                [default: hanning]
     Outputs:
        frqsFFT_Hz: grid of frequency for the interval 
                [0, Fs_Hz] (in Hz)
        auto-spectrum of x1
        auto-spectrum of x2
        cross-spectrum of (x1,x2)
        MSC (magnitude square coherence)
    """
    Fs_Hz       = float(Fs_Hz)
    N           = len(x1);
    sqrtLfft    = sqrt(Lfft);
    shiftSignal = int((1-overlapFFT)*Lfft);
    NblocksFFT  = int((N-(Lfft-shiftSignal))/shiftSignal);
    allFFTs11   = zeros([Lfft,NblocksFFT], dtype='complex');
    allFFTs22   = zeros([Lfft,NblocksFFT], dtype='complex');
    if smoothwindow == 'hanning':
        Hwin = hanning(Lfft, sym=False);
    elif smoothwindow == 'hamming':
        Hwin = hamming(Lfft, sym=False);
    else:
        Hwin  = hanning(Lfft, sym=False);
    #========= normalisation
    # not useful if only PSD ratios are considered
    Hwin = Hwin * sqrt(Lfft/dot(transpose(Hwin),Hwin));
    for ibF  in range(NblocksFFT):
        ibT  = range((ibF)*shiftSignal,Lfft+(ibF)*shiftSignal);
        x1_i = x1[ibT] * Hwin;
        x1_i = x1_i-mean(x1_i); 
        x2_i = x2[ibT] * Hwin;
        x2_i = x2_i-mean(x2_i);
        allFFTs11[:,ibF] = fft(x1_i,Lfft)/sqrtLfft;
        allFFTs22[:,ibF] = fft(x2_i,Lfft)/sqrtLfft;
        
    SDs11  = mean(abs(allFFTs11) ** 2,1);
    SDs22  = mean(abs(allFFTs22) ** 2,1);
    SDs12  = mean(allFFTs11 * conj(allFFTs22),1);
    MSC = (abs(SDs12) ** 2) / (SDs22 * SDs11);

    frqsFFT_Hz = array(range(Lfft))*Fs_Hz/Lfft;
    
    return frqsFFT_Hz, SDs11, SDs22, SDs12, MSC
    
#==============================================================    
def logMSCtheoGaussian(xsensor_m, 
   F_Hz,
   azimuth_deg,
   elevation_deg,
   velocity_mps,
   sigma_azimuth_deg,
   sigma_elevation_deg,
   sigma_velocity_mps):
    
    Lfft = len(F_Hz);
    F2_Hz = F_Hz * F_Hz;
    azimuth_rd = azimuth_deg*pi/180
    cosa = cos(azimuth_rd);
    sina = sin(azimuth_rd);
    elevation_rd = elevation_deg*pi/180
    cose = cos(elevation_rd);
    sine = sin(elevation_rd);
    c_mps = velocity_mps   
    
    sigma_azimuth_rd   = sigma_azimuth_deg*pi/180
    sigma_elevation_rd = sigma_elevation_deg*pi/180
       
    Saec2 = diag(array([sigma_azimuth_rd**2, 
                       sigma_elevation_rd**2, sigma_velocity_mps**2]))
    
    Jacobian = zeros([3,3])
    
#    array([-sin(a)*cos(e),-cos(a)*cos(e),sin(e)])/c

    Jacobian[:,0] = array([-cosa*cose, sina*sine, sina*cose/c_mps])/c_mps;
    Jacobian[:,1] = array([sina*cose, cosa*sine, cosa*cose/c_mps])/c_mps;
    Jacobian[:,2] = array([0, cose, -sine/c_mps])/c_mps;
    
    Gamma2_epsilon = dot(transpose(Jacobian), dot(Saec2,(Jacobian)));
       
    M = size(xsensor_m,0);
    C = M*(M-1)/2;
    logMSC = zeros([Lfft,C])
    cp = 0;
    for i1 in range(M-1):
        for i2 in range(i1+1,M):
            diffloc = xsensor_m[i2,:] - xsensor_m[i1,:];
            logMSC[:,cp] = -4*pi*pi*F2_Hz*(dot(dot(diffloc,Gamma2_epsilon),\
                  transpose(diffloc)))
            cp=cp+1
            
    return logMSC
    
#================================================
def generateISwithLOCgaussian(T_sec, Fs_Hz, xsensor_m,
                              azimuth0_deg,
                              elevation0_deg,
                              velocity0_mps,
                              sigma_azimuth_deg,
                              sigma_elevation_deg,
                              sigma_velocity_mps):
       
    M     = size(xsensor_m,0);
    N     = int(T_sec*Fs_Hz)
    x     = zeros([N,M]);
    Lruns = 200;    
    
    for il in range(Lruns):
        x0 = random.randn(N)       
        azimuth_deg   = azimuth0_deg+sigma_azimuth_deg*random.randn();
        elevation_deg = elevation0_deg+sigma_elevation_deg*random.randn();
        velocity_mps  = velocity0_mps+sigma_velocity_mps*random.randn();
        x_cur         = generateIS(x0, Fs_Hz, xsensor_m, azimuth_deg, 
                       elevation_deg, velocity_mps);
        x=x+x_cur
        
    x    = x/Lruns
    return x

#===================================================================
def alignmentwrt1(signal,startindex,windowlength_samples, 
                  rate = 2, visu = 0):
    """
# we align signals wrt to sensor 1
# with structured delays wrt the sensor locations
# That means that from the delays we extract the predicted delays
# with all combinations of sensors
#===================================================================
# [xalign, tkl_pts] = 
#    alignmentwrt1(signal,startindex,windowlength_samples)
#===================================================================
# Inputs:
#   - signal: array size (N,M)
#             N the observation number
#             M sensor number
#   - startindex : beginning index
#   - windowlength_samples :
#==
# Outputs:
#   - xalign : time-aligned signals
#
# Used functions
#   - delayedsignalF.m
#====================================================================
# last modified : 8/11
    
    """

    N         = size(signal,0);
    Nr        = N*rate
    signal    = resample(signal,int(Nr));
    windowlength_samples = rate*windowlength_samples;
    startindex = startindex*rate;
    
    M         = size(signal,1);
    tkl       = zeros(M);
    cormax    = zeros(M)
    xe        = signal[startindex:startindex+windowlength_samples,:];
    xt1       = xe[:,0];
    for ks  in range(1,M):
        xt2             = xe[:,ks];
        corkl           = correlate(xt1,xt2);
        indmaxkl        = argmax(corkl);
        taukl           = windowlength_samples - indmaxkl-1;
        # Parabolic interpolation
        yykl            = corkl[indmaxkl-1:indmaxkl+2];
        tcorrection,cormax[ks], alphakl = \
            parabolicinterpolation((-1.0,0.0,1.0),yykl)       
        tkl[ks] = (taukl-tcorrection); 
        
        if visu:
            nptsparabole = 31;
            xpar = -1.0\
             +2.0*array(range(nptsparabole))/(nptsparabole-1.0);
            
            parabole = zeros(nptsparabole)
            for ip in range(nptsparabole):
                parabole[ip] = \
                dot(array([1, xpar[ip], xpar[ip] ** 2]),alphakl);
            HorizontalSize = 8
            VerticalSize   = 4
            
            plt.figure(num=1, figsize=(HorizontalSize,VerticalSize))
            plt.plot(corkl, '.-')
            plt.hold('True')
#                  plt.plot(N-tkl[idC]-1, max_corr[idC], 'or')
            plt.plot(windowlength_samples-(tkl[ks]+1), cormax[ks], 'or')
            plt.plot(windowlength_samples-(taukl-xpar+1),parabole,'-r')
            plt.plot(960*array([1.0, 1.0]),array([-0.2, 0.5]),':k')
            
            plt.xlim(indmaxkl+array([-3, 3]));
            plt.hold('False')
            plt.show()

    
    #==
    xalign                  = zeros([windowlength_samples,M]);
    for ss in range(M):
        xalign[:,ss]        = delayedsignalF(xe[:,ss],-tkl[ss])
    
    tkl_pts = tkl/rate;
    xalign = resample(xalign,N);
    return xalign, tkl_pts

#====================================================================
def consistence(signal, rate=1, visu=0, corrflag = 0):
#====================================================================
    """
# Synopsis:
#    consistence, tkl = consistence(signal, rate=1, visu=0, corrflag = 1)       
#=========
#Inputs:
#   - signal: array size (N,M) where
#             N is the observation number
#             M is the sensor number
#   - rate: interpolation rate
#   - corrflag: if true compute correlation, if false compute covariance
#   - visu: display the graphic around the maximum
#=========
#Outputs:
#   - consistence
#   - tkl: sequance of delays, array of length M*(M-1)/2
#=========
# Used functions
#   - correlate (from scipy.signal)
#   - resample (from scipy.signal)
   """
    nbsensors = size(signal,1);
    N         = size(signal,0);   
    N         = N*rate
    signal    = resample(signal,int(N));
      
    Combi                   = nbsensors*(nbsensors-1)/2;
    tkl                     = zeros(Combi);
    tkl2D                   = zeros([nbsensors,nbsensors]);
    max_corr                = zeros(Combi);
    idC                     = -1;
    for ks in range(nbsensors-1):
        for ls in range(ks+1,nbsensors):
            idC             = idC + 1;
            corkl           = correlate(signal[:,ks],signal[:,ls]);
            if corrflag:
                corkl = corkl / sqrt(sum(signal[:,ks]**2)*sum(signal[:,ls]**2))
                
            indmaxkl        = argmax(corkl);
            taukl           = N - indmaxkl-1;
            # interpolation parabolique
            if (indmaxkl<2) | (indmaxkl>2*N-2):
                consistence = nan;
                tkl         = nan;
            else:
                yykl            = corkl[indmaxkl-1:indmaxkl+2];
                invH            = array([[0.0, 1.0, 0.0], \
                        [-0.5, 0.0, 0.5], \
                        [0.5, -1.0, 0.5]])
                alphakl         = dot(invH,yykl)
                # we perform index in 1:Combi
                tkl[idC]         = (taukl + alphakl[1]/2.0/alphakl[2]);
                tkl_Fs1          = -alphakl[1]/2/alphakl[2]
                max_corr[idC]    = alphakl[0] + alphakl[1]*tkl_Fs1 \
                         + tkl_Fs1*tkl_Fs1*alphakl[2];
                if visu :# tkl(idC)<-100
                    nptsparabole = 31;                    
                    xpar = -1.0\
                     +2.0*array(range(nptsparabole))/(nptsparabole-1);
                    parabole = zeros(nptsparabole)
                    for ip in range(nptsparabole):
                        parabole[ip] = \
                        dot(array([1, xpar[ip], xpar[ip] ** 2]),alphakl);
                    
                    HorizontalSize = 8
                    VerticalSize   = 34
                    
                    plt.figure(num=1, figsize=(HorizontalSize,VerticalSize))

                    plt.subplot(Combi,1,idC+1)
                    plt.plot(corkl, '.-')
                    plt.hold('True')
                    plt.plot(N-(tkl[idC]+1), max_corr[idC], 'or')
                    plt.plot(N-(taukl-xpar+1),parabole,'-r')
                    plt.plot(960*array([1.0, 1.0]),array([-0.2, 0.5]),':k')
                    plt.hold('False')
                    plt.xlim(indmaxkl+array([-5, 5]));
                    plt.draw()
                    
                tkl2D[ks,ls] = tkl[idC];
                tkl2D[ls,ks] = -tkl2D[ks,ls];
    tkl = tkl/rate;
    tkl2D = tkl2D/rate;
    consistence2 = 0;
    c3=0;
    for kk in range (nbsensors-1):
        for jj in range(kk+1,nbsensors):
            for mm in range(jj+1,nbsensors):
                increm = abs(tkl2D[kk,jj]+tkl2D[jj,mm]+tkl2D[mm,kk]) **2;
                consistence2 = consistence2 + increm;
                c3=c3+1;
    consistence = sqrt(consistence2/c3)
    return consistence, tkl

#====================================================================
def MCCM(signal, rate = 1, corrflag = 1):
#=====================================================================
    """
# Mean of Cross-Covariance Maximum
#    synopsis: MCCM = MCCM(signal, rate = 1, corrflag = 1)
#=========
#Inputs:
#   - signal: array size (N,M) where
#             N the observation number
#             M sensor number
#   - rate: interpolation rate
#   - corrflag: if true compute correlation, if false compute covariance
#=========
# Outputs:
#   - MCCM
#
#=========
# Used functions
#   - correlate (from scipy.signal)
#   - resample (from scipy.signal)
    """
    nbsensors = size(signal,1);
    N         = size(signal,0);   
    N         = N*rate
    signal    = resample(signal,int(N));
 
    signal_cc = zeros([size(signal,0),size(signal,1)]);
    for im in range(nbsensors):
        signal_cc[:,im] = signal[:,im]-mean(signal[:,im]);
    
    Combi = nbsensors*(nbsensors-1)/2;
    tkl = zeros(Combi);
    max_corr = zeros(Combi);
    idC = -1;
    for ks in range(nbsensors-1):
        for ls in range(ks+1,nbsensors):
            idC             = idC + 1;
            corkl           = correlate(signal_cc[:,ks],signal_cc[:,ls]);
            if corrflag:
             corkl           = corkl / \
                  sqrt(sum(signal_cc[:,ks]**2)*sum(signal_cc[:,ls]**2))

            indmaxkl        = argmax(corkl);
            taukl           = N - indmaxkl-1;
            # interpolation parabolique
            if (indmaxkl<2) | (indmaxkl>2*N-2):
                MCCM = nan;
                tkl = nan;
            else:
                yykl             = corkl[indmaxkl-1:indmaxkl+2];
#                invH             = array([[0.0, 1.0, 0.0], \
#                        [-0.5, 0.0, 0.5], \
#                        [0.5, -1.0, 0.5]])
#                alphakl          = dot(invH,yykl)
#                # we perform index in 1:Combi
#                tkl[idC]         = (taukl + alphakl[1]/2.0/alphakl[2]);
#                tkl_Fs1          = -alphakl[1]/2/alphakl[2]
#                max_corr[idC]    = alphakl[0] + alphakl[1]*tkl_Fs1 \
#                         + tkl_Fs1*tkl_Fs1*alphakl[2];
#                         
#
#                # Parabolic interpolation
                tcorrection,cormaxP, alphaklP = \
                    parabolicinterpolation((-1.0,0.0,1.0),yykl)       
                tkl[idC] = (taukl-tcorrection); 
                max_corr[idC] = cormaxP

    MCCM = mean(max_corr)/rate;
    return MCCM, tkl/rate

#=====================================================================
#=====================================================================
##%
#    listevents = {};    
#    for ik in range(nb_events):
#        events  = struct(sensorinfo=struct(coordinate=xsensor_m, name="IS37"),
#                realSOI   = struct(flag = 0, database = "filename"),
#                realNOISE = struct(flag = 0, database = "filename"),
#                SOIazimuth_deg   = SOI_azimuths_deg[ik],
#                SOIelevation_deg = SOI_elevations_deg[ik],
#                SOIvelocity_mps  = SOI_velocities_mps[ik],
#                SOIband_Hz       = SOI_Flims_Hz[ik,:],
#                SOITOA_s         = SOI_TOA_sec[ik],
#                SOIduration_s    = SOI_dur_sec[ik],
#                SOIdelay_sec       = SOI_delays_sec[:,ik]##-SOI_delays_sec[0,ik],
#                SOISNR_dB        = -20*log10(SOI_sigma_noise[:,ik]),
#                emergent         = struct(flag=flag_emergent), 
#                HugeNoise        = struct(flagHugeNoise=flag_hugeNoise,
#                                          params = paramsHugeNoise,
#                            sensor_HugeNoise = sensor_HugeNoise[ik]),
#                phase_distorsion = struct(flagdistorsion=flagphasedistortion, 
#                                         direct=Directallpass, reverse=Reverseallpass,
#                                         numsensor=mselect_distphase),
#                SON              = struct(flagSON=flagSON, parameters=paramsSON),
#                LOC              = struct(flagLOC=flagLOC, parameters=paramsLOC),
#                failure          = struct(flagfailure=flagfailure, sensor=sensor_failures[ik],
#                                          probability=probability)
#                )
#        listevents[ik] = events
        
#%%
#def synthetizer(sensors, Fs_Hz, totalTime_s, nb_events, 
#                         realSOI,realNOISE,SOIdurationrange_s, 
#                         SOIfrequencywidth_Hz_Hz, SOISNRrange_dB, 
#                         SOIazimuthList_deg, 
#                         SOIelevationList_deg, SOIvelocityList_mps,
#                         flagTimeoverlap, emergent, LOC, SON, hugeNoise,
#                         failure,flagphasedistortion):
def synthetizer(sensors, SOI, LOC, SON, 
                          hugeNoise, failure, emergent):                             
#=====================================================================
# SYNOPSIS:
#  record = 
#     synthetizer( 
#     sensors_m, 
#     SOI,
#     LOC, 
#     SON, 
#     hugeNoise, 
#     failure,
#     emergent)
#
#=====================================================================
# Inputs:
#     sensors: structure
#        .name = sensor name
#        .data = geoloc structure M by 5
#          latitude_deg,longitude_deg,
#          elevation_km, X_km,Y_km
#     SOI = struct(name = 'soiinfo', flag_SOItrue = flag_SOItrue,
#             soidatabase = SOI_database,
#             noiseflag_real = NOISEflag_real,
#             noise_database = NOISE_database,
#             totaltime_s = totalTime_s,
#             Fs_Hz = SOIFs_Hz, 
#             nb_events = SOInb_events,
#             durationrange_s = SOIdurationrange_s,
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
#         M: number of sensors
#     .events: structure of length NB_EVENTS
#       sensors: structure
#        .name = sensor name
#        .data = geoloc structure M by 5
#          latitude_deg,longitude_deg,
#          elevation_km, X_km,Y_km
#       SOI = struct(name = 'soiinfo', flag_SOItrue = flag_SOItrue,
#             soidatabase = SOI_database,
#             noiseflag_real = NOISEflag_real,
#             noise_database = NOISE_database,
#             totaltime_s = totalTime_s,
#             Fs_Hz = SOIFs_Hz, 
#             nb_events = SOInb_events,
#             durationrange_s = SOIdurationrange_s,
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
#=====================================================================
    Fs_Hz                       = SOI.Fs_Hz;
    totalTime_s                 = SOI.totaltime_s
    
    M                           = len(sensors);
    xsensors_m                  = zeros([M,3])
    for im in range(M):
        ism = sensors[im];
        xsensors_m[im,:] = \
           1000.0*array([ism.geolocs.X_km, ism.geolocs.Y_km, ism.geolocs.Z_km])
    
    nb_events                   = SOI.nb_events;

    flagSOItrue                 = SOI.flag_SOItrue;
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
    
    SOI_durationrange_s         = SOI.durationrange_s
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
    T_pts              = int(totalTime_s*Fs_Hz);
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
    SOI_durationmin_s = SOI_durationrange_s[0];
    SOI_durationmax_s = SOI_durationrange_s[1];
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
        meanlength_sec = (IG_totattimeMUL*totalTime_s\
                           -SOI_durationmax_s)/nb_events;
        SOI_TOA_sec[0] = meanlength_sec * random.rand();
        SOI_dur_sec[0] = (SOI_durationmax_s-SOI_durationmin_s)\
                       *random.rand()+SOI_durationmin_s;
        SOI_dur_sec[0] = max([SOI_dur_sec[0], SOI_duration_at_least_sec]);
        nexteventbegin = SOI_TOA_sec[0]+SOI_dur_sec[0];
        for ik in range (1,nb_events):
            SOI_TOA_sec[ik] = meanlength_sec\
                     * random.rand()+nexteventbegin;
            SOI_dur_sec[ik] = (SOI_durationmax_s-SOI_durationmin_s)\
                     * random.rand()+SOI_durationmin_s;
            SOI_dur_sec[ik] = max([SOI_dur_sec[ik], \
                       SOI_duration_at_least_sec]);
            nexteventbegin  = SOI_TOA_sec[ik]+SOI_dur_sec[ik];
            if nexteventbegin > totalTime_s-SOI_durationmax_s:
                print('too many events for the total duration');
                return []

    else:
        SOI_TOA_sec = IG_totattimeMUL*totalTime_s\
                 * random.rand(nb_events);
        SOI_dur_sec = (SOI_durationmax_s-SOI_durationmin_s)\
                 *random.rand(nb_events)+SOI_durationmin_s;
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
    # randomly selected sensors for huge noise and failure
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
    #       flagSOItrue=0:
    #       either as second order filtering of white noise
    #       flagSOItrue=1:
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
        if flagSOItrue:
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
     
#    [bid, indsortTOA] = sort([events(:).SOITOA_s]);
#    events            = events(indsortTOA);
#    
#    #====== we keep only event TOA less than totalTime_s
#    ikMAX             = [events(:).SOITOA_s]<totalTime_s;
#    nbsamples         = fix(totalTime_s*Fs_Hz);
#    record.signals    = observationswithnoise(1:nbsamples,:);
#    record.SOIonky    = SOIonly(1:nbsamples,:);
#    record.events     = events(ikMAX);

#=====================================================================
#=====================================================================
#=====================================================================
#function [signal_withLOC, theta0_spm, Gamma2_epsilon] = 
#    genelwwLOCwithoutdelay(one_signal,
#    orderLPC,
#    Fs_Hz, az_deg,el_deg, velocity_mps, Sigma_aec, xsensor_m)
def genelwwLOCwithoutdelay(one_signal, orderLPC,
    Fs_Hz, az_deg,el_deg, velocity_mps, Sigma_aec, xsensors_m):
#=====================================================================
# generate loss of coherence with noise
# signal_out = generateCoherenceN( 
#     duration_s, Fs_Hz, aec, sigmaaec, xsensor_m)
# Inputs
#  ones_signal    : one SOI
#  orderLPC     :
#  Fs_Hz     : sampling frequency in Hz
#  az_deg,el_deg, velocity_mps:
#         deterministic part of the DOA
#
#  Sigma_aec : random part of the DOA
#
#  xsensor_m: M x 3
#     3D locations of the M sensors
#
#=======
# Output
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
# Gamma2_epsilon:
#          positive matrix en s2/m2Ctrl+S (LOC)
#=====================================================================
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
#%%
##=====================================================================
##=====================================================================
#function sigOUT = readDATABASE(database,Nlong,Fs_Hz)
#xx = load(database);
#nam = fieldnames(xx);
#eval(sprintf('sigIN = xx.#s;',nam{1}));
#N = length(sigIN);
#startS = fix((N-Nlong-1)*rand);
#sigOUT = sigIN(startS+(1:Nlong));
# Fmin_Hz = 0.1;
# Fmax_Hz = 5;
# [bb,aa] = butter(4,2*[Fmin_Hz Fmax_Hz]/Fs_Hz);
# sigOUT = filter(bb,aa,sigIN);
#=====================================================================

#===============================================
#    function [pz, meanpz, varpz] = ...
#    asymFalpha(f,alpha,N1,m0,m1,sigma20,sigma21)
def asymFalpha(f,alpha,N1,m0,m1,sigma20,sigma21):
#====================================================
# Asymptotic probability density function (PDF)
# of the Falpha-score:
#
#   Falpha-score^-1 = 
#        alpha Precision^-1 + (1-alpha) Recall^-1
#   
#   Precision = TP/(TP+FP)
#   Recall    = TP/(TP+FN)
#
# Assumptions:
#      Y_0(1),...,Y_0(N0) \sim iid Binomial
#          with Pr(Y_0(k)=1)=pi_0
#      Y_1(1),...,Y_1(N0) \sim iid Binomial
#          with Pr(Y_1(k)=1)=pi_1
#      TP = sum_k Y_1(k)
#      FP = N0-sum_k Y_0(k)
#      FN = N1-sum_k Y_1(k)
#======
# Synopsis:
#  pz,mean1Fscoreth, sigma2Fscoreth = ...
#    asymFscore(f,N1,m0bar,m1,sigma20,sigma21)
#
#======
# Inputs:
#    - f : F-alpha value
#    - N1 : number of positive data
#    - m0bar : mean of the FP, typically N0-N0*pi_0
#    - m1 : mean of the TP, typically N1*pi_1
#    - sigma20 : variance of the TN, 
#             typically N0*pi_0(1-pi_0)
#    - sigma21 : variance of the TP, 
#             typically N1*pi_1(1-pi_1)
#======
# Outputs
#    - value of the PDF
#    - meanpz : mean
#    - varpz : variance
#======
# used functions
#    - INTEGRAL not available on R2011 replaced by
#      QUADGK with infinities replaced by 5!
#====================================================
    m0=float(m0)
    m1=float(m1)
    sigma20=float(sigma20)
    sigma21=float(sigma21)
    f=float(f)

    pz = asympdfFalpha(f, alpha, N1, m0,m1,sigma20,sigma21);
#==========
    infini = inf;
#==========
    meanpz, bid = quad(xpdfFalpha, -infini,infini, args = (alpha,N1,m0,m1,sigma20,sigma21));
    m2pz, bid = quad(x2pdfFalpha, -infini,infini, args = (alpha,N1,m0,m1,sigma20,sigma21));
    varpz  = m2pz-meanpz**2;
    return pz, meanpz, varpz
    
def xpdfFalpha(f,alpha,N1,m0,m1,sigma20,sigma21):
    xpdf = f * asympdfFalpha(f,alpha,N1,m0,m1,sigma20,sigma21)
    return xpdf
def x2pdfFalpha(f,alpha,N1,m0,m1,sigma20,sigma21):
    f2 = f**2
    x2pdf = f2 * asympdfFalpha(f,alpha,N1,m0,m1,sigma20,sigma21)
    return x2pdf
#====================================================
#function pz = asympdfFalpha(f,alpha,N1,m0,m1, ...
#    sigma20,sigma21)
def asympdfFalpha(f,alpha,N1,m0,m1,sigma20,sigma21):
    m0=float(m0)
    m1=float(m1)
    sigma20=float(sigma20)
    sigma21=float(sigma21)
    f=float(f)
    
    ma      = (1.0-alpha)*N1+alpha*m0;
    sigma2a = alpha*alpha*sigma20;
    sigmaa  = sqrt(sigma2a);
    sigma1  = sqrt(sigma21);
    if f==0:
        uont     = ma/sigmaa
        c2uvont3 = sigma2a*uont * exp(-(m1*m1)/2.0/sigma21)
        aux1     = (c2uvont3/sigmaa/sigma1/sqrt(2.0*pi))*(2.0*norm.cdf(uont)-1.0);
        aux2     = exp(-ma*ma/sigma2a/2.0-m1*m1/sigma21/2.0);
        aux3     = aux2*sigmaa/pi/sigma1        
        pz       = aux1+aux3
    else:
        z       = (1.0 / f) - alpha;
        tz      = sqrt((z * z)/sigma2a+1.0/sigma21);
        uz      = ma*z/sigma2a+m1/sigma21;
        vz      = exp(0.5*(uz * uz) / (tz * tz)-ma*ma/sigma2a/2-m1*m1/sigma21/2.0);
        aux1    = (uz * vz) / (tz ** 3)\
               / sqrt(2.0*pi*sigma2a*sigma21);
    
        aux2    = pi * sqrt(sigma2a*sigma21) * (tz * tz);
        aux3    = exp(-ma*ma/sigma2a/2.0-m1*m1/sigma21/2.0);
        pzb     = aux1 * (2.0 * norm.cdf(uz / tz)-1.0) + aux3 / aux2;
        pz      = pzb / (f ** 2);
    return pz
#====================================================
#function [pz, meanpz, varpz] = ...
#    asymPrecision(u,m0bar,m1,sigma20,sigma21)
def asymPrecision(f,m0,m1,sigma20,sigma21):
#====================================================
# Asymptotic probability density function (PDF)
# of the Precision:
#      Precision = TP/(TP+FP)
#
# Assumptions:
#      Y_0(1),...,Y_0(N0) \sim iid Binomial
#          with Pr(Y_0(k)=1)=pi_0
#      Y_1(1),...,Y_1(N0) \sim iid Binomial
#          with Pr(Y_1(k)=1)=pi_1
#      TP = sum_k Y_1(k)
#      FP = N0-sum_k Y_0(k)
#======
# Synopsis:
#  [pz, meanpz, varpz] = ...
#     asympdfPrecision(u,m0bar,m1,sigma20,sigma21)
#
#======
# Inputs:
#    - u  : Precision value
#    - m0bar : mean of the FP, typically N0-N0*pi_0
#    - m1 : mean of the TP, typically N1*pi_1
#    - sigma20 : variance of the TN, 
#             typically N0*pi_0(1-pi_0)
#    - sigma21 : variance of the TP, 
#             typically N1*pi_1(1-pi_1)
#======
# Output
#    - value of the PDF
#    - meanpz : mean
#    - varpz : variance
#======
# used functions
#    - INTEGRAL not available on R2011 replaced by
#      QUADGK with infinities replaced by 1!
#
#====================================================
    m0=float(m0)
    m1=float(m1)
    sigma20=float(sigma20)
    sigma21=float(sigma21)
    f=float(f)
 
    pz = asympdfPrecision(f,m0,m1,sigma20,sigma21);
    infini = 1000;
    
    meanpz, bid = quad(xpdfPrecision, -infini,infini, args = (m0,m1,sigma20,sigma21));
    m2pz, bid = quad(x2pdfPrecision, -infini,infini, args = (m0,m1,sigma20,sigma21));
    
    varpz  = m2pz-meanpz**2;
    
    return pz, meanpz, varpz
    
def xpdfPrecision(f,m0,m1,sigma20,sigma21):
    xpdf = f * asympdfPrecision(f,m0,m1,sigma20,sigma21)
    return xpdf
def x2pdfPrecision(f,m0,m1,sigma20,sigma21):
    f2 = f**2
    x2pdf = f2 * asympdfPrecision(f,m0,m1,sigma20,sigma21)
    return x2pdf
    
#===============================================================
def asympdfPrecision(u,m0,m1,sigma20,sigma21):
    
    m0=float(m0)
    m1=float(m1)
    sigma20=float(sigma20)
    sigma0 = sqrt(sigma20)
    sigma21=float(sigma21)
    sigma1 = sqrt(sigma21)
    u=float(u)
    
    if u==0:
        uont     = m0/sigma0
        c2uvont3 = sigma20*uont * exp(-(m1*m1)/2.0/sigma21)
        aux1     = (c2uvont3/sigma0/sigma1/sqrt(2.0*pi))*(2.0*norm.cdf(uont)-1.0);
        aux2     = exp(-m0*m0/sigma20/2.0-m1*m1/sigma21/2.0);
        aux3     = aux2*sigma0/pi/sigma1   
        pz       = aux1+aux3
    else:
        z  = (1 / u) - 1;
        tz = sqrt((z * z)/sigma20+1.0 /sigma21);
        uz = m0*z/sigma20+m1/sigma21;
        vz = exp(0.5*( uz * uz) / (tz * tz)\
             - m0*m0/sigma20/2-m1*m1/sigma21/2 );
    
        aux1 = (uz * vz) / (tz ** 3) \
            / sqrt(2*pi*sigma20*sigma21);
    
        aux2 = pi * sqrt(sigma20*sigma21) * (tz * tz);
        d    = exp(-m0*m0/sigma20/2.0-m1*m1/sigma21/2.0);
        pzb  = aux1 * (2*norm.cdf(uz / tz)-1) + d / aux2;
        pz   = pzb / (u ** 2);
    return pz
#===============================================================
#function [pz, meanpz, varpz] = ...
#    asymRecall(r,N1,m1,sigma21)
def asymRecall(r,N1,m1,sigma21):
#===============================================================
# Asymptotic probability density function (PDF)
# of the Recall:
#   Recall = TP/(TP+FN) = TP/N1
#
# Assumptions:
#      Y_1(1),...,Y_1(N0) \sim iid Binomial
#          with Pr(Y_1(k)=1)=pi_1
#      TP = sum_k Y_1(k)
#      FN = N1-sum_k Y_1(k)
#======
# Synopsis:
#  [pz, meanpz, varpz] = asympdfRecall(r,N1,m1,sigma21)
#
#======
# Inputs:
#    - r : Recall value
#    - N1 : number of positive data
#    - m1 : mean of the TP, typically N1*pi_1
#    - sigma21 : variance of the TP, 
#             typically N1*pi_1(1-pi_1)
#======
# Output
#    - value of the PDF
#    - meanpz : mean
#    - varpz : variance
#===============================================================
    r=float(r)
    m1=float(m1)
    sigma21 = float(sigma21)
    sz     = sigma21/N1/N1;
    pz     = exp(-0.5*((r-m1/N1) **2)/sz)/sqrt(2.0*pi*sz);
    meanpz = m1/N1;
    varpz = sz;
    return pz, meanpz, varpz
#===============================================================
def rocauc(valH0, valH1, alpha_percent, nbbins=20, nbboot=300, scale='log'):
    """
    Performs the roc curve of two sequences and the area under
    the roc (auc)
    Inputs:  valH0, valH1, two sequences of any length
             nbbins number of points of the roc
    Outputs: alpha=false alarm, beta=detection probability, auc
    
    synopsis:
    alpha, beta, CIalpha, CIbeta, eauc, std_eauc_exp, 
              std_eauc_boot = 
              rocauc(valH0, valH1, alpha_percent, 
              nbbins=20, nbboot=300)
    """  
    N0 = len(valH0);
    N1 = len(valH1);
    smax = max([max(valH0),max(valH1)])
    smin = min([min(valH0),min(valH1)])
    if scale=='log':
        ranges = logspace(log10(smin),log10(smax),nbbins)
    else:
        ranges = linspace(smin,smax,nbbins)
    alpha = zeros(nbbins);
    beta = zeros(nbbins);
    for i in range(nbbins):
        alpha[i] = float(sum(valH0>ranges[i]))/N0;
        beta[i] = float(sum(valH1>ranges[i]))/N1;

    #== CI from the binomial variance
    calpha       = norm.ppf(1.0-(1.0-alpha_percent/100.0)/2.0);
    calpha2      = calpha*calpha;
    
    CIalpha      = zeros([nbbins,2])
    CIbeta       = zeros([nbbins,2])
    aux0         = calpha*sqrt((alpha * (1.0-alpha))/N0\
               +calpha2/4.0/N0/N0);
    den0         = 1/(1+calpha2/N0);
    CIalpha[:,0] = (alpha + (calpha2/2.0/N0)-aux0)*den0;
    CIalpha[:,1] = (alpha + (calpha2/2.0/N0)+aux0)*den0;
    
    aux1         = calpha*sqrt((beta * (1.0-beta))/N1 \
             +calpha2/4.0/N1/N1);
    den1         = 1/(1+calpha2/N1);
    CIbeta[:,0]  = (beta + calpha2/2.0/N1-aux1)*den1;
    CIbeta[:,1]  = (beta + calpha2/2.0/N1+aux1)*den1;

    eauc = aucW(valH0,valH1)
    
    #===== std of the estimate
    # exponential model
    eauc2 = eauc*eauc;
    B0 = (2.0*eauc2) / (1.0+eauc);
    B1 = eauc/ (2.0-eauc);
    var_eauc = eauc2+eauc-(N0+N1)*eauc2+(N0-1.0)*B0+(N1-1.0)*B1;
    std_eauc_exp = sqrt(var_eauc / (N0*N1));
    # bootstrap
    eauc_boot = zeros(nbboot);
    for ib in range(nbboot):
        ind0 = random.randint(1,N0,N0);
        ind1 = random.randint(1,N1,N1);
        eauc_boot[ib] = aucW(valH0[ind0],valH1[ind1]);
    
    std_eauc_boot     = std(eauc_boot);
    
    return alpha, beta, CIalpha, CIbeta, eauc, std_eauc_exp, \
               std_eauc_boot
#===============================================================
def aucW(valH0,valH1):
    
    N0 = len(valH0);
    N1 = len(valH1);
    W = 0;
    if N0<=N1:
        for i0 in range(N0):
            W = W+sum((valH1>valH0[i0])+0.5*(valH1==valH0[i0]));
    else:
        for i1 in range(N1):
            W = W+sum((valH1[i1]>valH0)+0.5*(valH1[i1]==valH0));
       
    W = float(W)/(N0*N1);
    return W

#===============================================================
def evalCRBwithgaussianLOC(xsensors_m, sigma2noise, aec, std_aec, 
                           duration_sec, Fs_Hz):
    
    """ Calculation of the Cramer-Rao bound 
    # synopsis:
    #    CRB, Jacobav =
    #     evalCRBwithLOC(xsensors_m, sigma2noise, C, polarparam, fK_Hz)
    # Inputs:
    #    xsensors_m : sensor locations in meter
    #    sigma2noise : noise variance
    #    aec : structure 
    #    if D=2  aec.a_deg
    #            aec.v_mps
    #    if D=3  aec.a_deg
    #            aec.e_deg
    #            aec.c_mps
    #    duration_sec: time duration in sec
    #    Fs_Hz: sampling frequency in Hz
    # Outputs:
    #    CRB, Jacobav, C0
    """
    M = size(xsensors_m,0);
    D = size(xsensors_m,1);
    N = int(duration_sec * Fs_Hz)
    K = N/2-1;
    fullfrequency_Hz = arange(N)*Fs_Hz/N;
#    deltaf = Fs_Hz/N;
    frequency_Hz = fullfrequency_Hz[range(1,K+1)]
    freq2_Hz = frequency_Hz ** 2;
    slowness_spm = zeros(D);
    
    if D==2:
        cosa  = cos(aec.a_deg*pi/180);
        sina  = sin(aec.a_deg*pi/180);
        v_mps = aec.v_mps;
        slowness_spm[0] = cosa/v_mps;
        slowness_spm[1] = sina/v_mps;
        delay_sec = dot(xsensors_m, slowness_spm);
        Jacobav = array([[-sina/v_mps, -cosa/v_mps/v_mps],
            [cosa/v_mps, -sina/v_mps/v_mps]]);
        Sigma_aec  = diag([std_aec.a_deg*pi/180.0, std_aec.c_mps]);
        Sigma2_aec = Sigma_aec * Sigma_aec;
        Sigma2_theta_spm = dot(dot(Jacobav,Sigma2_aec),Jacobav.transpose());

    elif D == 3:
        a_rd = aec.a_deg*pi/180
        e_rd = aec.e_deg*pi/180
        cosa  = cos(a_rd);
        sina  = sin(a_rd);
        cose  = cos(e_rd);
        sine  = sin(e_rd);
        c_mps = aec.c_mps;
        # for 2D
        v_mps = c_mps/cose;
        
        slowness_spm = aec2theta(a_rd,e_rd,c_mps)
        delay_sec = dot(xsensors_m, slowness_spm);        

        # slowness_spm[0] = -sin(a_rd)*cos(e_rd)/c_mps
        # slowness_spm[1] = -cos(a_rd)*cos(e_rd)/c_mps
        # slowness_spm[2] = sin(e_rd)])/c_mps
        Jacobaec = array([
            [-cosa*cose/c_mps, sina*sine/c_mps, sina*cose/c_mps/c_mps],
            [sina*cose/c_mps, cosa*sine/c_mps, cosa*cose/c_mps/c_mps],
            [0.0, cose/c_mps, -sine/c_mps/c_mps]]);

        # as v=c/cos(e)
        # slowness_spm[0] = cosa/v;
        # slowness_spm[1] = sina/v;
        # slowness_spm[2] = tane/v;
        Jacobav = array([[-cosa/v_mps, sina/v_mps/v_mps, 0],
                [sina/v_mps, cosa/v_mps/v_mps, 0],
                [0, -sine/cose/v_mps/v_mps, 1.0/cose/cose/v_mps]])
        
        Sigma_aec  = diag([std_aec.a_deg*pi/180.0, std_aec.e_deg*pi/180.0, std_aec.c_mps]);
        Sigma2_aec = Sigma_aec * Sigma_aec;
        Sigma2_theta_spm = dot(dot(Jacobaec,Sigma2_aec),Jacobaec.transpose());

    C = ones([K,M,M]);
    for im1 in range(M-1):     #=1:M-1
        for im2 in range(im1+1,M):    #=im1+1:M
            diffloc_m = xsensors_m[im2,:]-xsensors_m[im1,:];
            logcoh_cp = -2*pi*pi*freq2_Hz * \
               dot(dot(diffloc_m.reshape(1,D),Sigma2_theta_spm),
                   diffloc_m.reshape(D,1));            
            C[:,im1,im2] = exp(logcoh_cp);
            C[:,im2,im1] = C[:,im1,im2];
            
    IM = eye(M);
    DK = exp(-2j*pi*dot(frequency_Hz.reshape(K,1),delay_sec.reshape(1,M)));
    FIM_ik = zeros([K,D,D]);
    oneoneT = dot(ones([M,1]),ones([1,M]));
    Gammam1drond = zeros([M,M,D], dtype=complex)
    
    for ik in range(K):
        D_ik = DK[ik,:];
        diagD_ik = diag(D_ik);
        if any(isnan(C)):
            C_ik = oneoneT;
        else:
            C_ik = C[ik,:,:];
        
        Gamma_ik = dot(dot(diagD_ik,C_ik),diagD_ik.conj())\
                   +sigma2noise*IM;
        invGamma_ik = inv(Gamma_ik)
        repD_ik = kron(ones(D),D_ik.reshape(M,1));
        # mult. term/term
        d_ik = -2j*pi*frequency_Hz[ik]*(xsensors_m*repD_ik);

        for ell in range(D):
            diagell = diag(d_ik[:,ell]);
            dGamma_ik_ellplus = dot(dot(diagell, C_ik),conj(diagD_ik));
            dGamma_ik_ellmoins = dot(dot(diagD_ik,C_ik),conj(diagell));
            dGamma_ik_ell = dGamma_ik_ellplus+dGamma_ik_ellmoins;
                        
            Gammam1drond[:,:,ell] = dot(invGamma_ik,dGamma_ik_ell);
       
        for q in range(D):
            for qp in range(D):
                FIM_ik[ik,q,qp] = real(trace(dot(Gammam1drond[:,:,q],
                            Gammam1drond[:,:,qp])))
        
#            for ellp in range(D):
#                diagellp = diag(d_ik[:,ellp]);
#                dGamma_ik_ellpplus = dot(dot(diagellp,C_ik),diagD_ik.conj());
#                dGamma_ik_ellpmoins = dot(dot(diagD_ik,C_ik),diagellp.conj());
#                dGamma_ik_ellp = dGamma_ik_ellpplus+dGamma_ik_ellpmoins;
#                auxFIMp = dot(invGamma_ik,conj(dGamma_ik_ellp.transpose()));
#                traceaux = trace(dot(auxFIM ,auxFIMp))
#                FIM_ik[ik,ell,ellp] = traceaux.real ;#0.5*traceaux.real;
    

    FIMslowness = sum(FIM_ik,axis = 0);# * deltaf ;    
    CRB = struct(slowness = pinv(FIMslowness))
    
    # in 3D
    # slowness_spm[0] = -sina*cose/c;
    # slowness_spm[1] = -cosa*cose/c;
    # slowness_spm[2] = sine/c;
    if D == 2:
        invJacobav = inv(Jacobav)
        CRB.av = dot(dot(invJacobav,CRB.slowness),
                     invJacobav.transpose());
    elif D==3:
        invJacobaec = inv(Jacobaec);
        CRB.aec =  dot(dot(invJacobaec,CRB.slowness),\
              invJacobaec.transpose());
        invJacobav = inv(Jacobav);
        CRBaux = dot(dot(invJacobav,CRB.slowness), invJacobav.transpose())
        CRB.av = CRBaux[0:2,0:2];
    
    if any(std_aec)==0:
        C0 = 1
    else:
        C0 = C
    return CRB, Jacobav, C0

#========================================
def CRBonazimuthonlywithoutLOC(xsensors_m, sigma2noise, aec,
                           duration_sec, Fs_Hz):
                               
    """ Calculation of the Cramer-Rao bound 
    when all slowness components are known
    EXCEPT the azimuth
    # synopsis:
    #    CRBaz = CRBonazimuthonlywithoutlOC
    #     (xsensors_m, sigma2noise, C, polarparam, fK_Hz)
    # Inputs:
    #    xsensors_m : sensor locations in meter
    #    sigma2noise : noise variance
    #    aec : structure 
    #            aec.a_deg
    #            aec.e_deg
    #            aec.c_mps
    #    duration_sec: time duration in sec.
    #    Fs_Hz: sampling frequency in Hz
    # Outputs:
    #    CRB of the azimuth
    """
    
    M = size(xsensors_m,0);
    
    D = size(xsensors_m,1);
    N = int(duration_sec * Fs_Hz)
    K = N/2-1;
    fullfrequency_Hz = arange(N)*Fs_Hz/N;
    frequency_Hz = fullfrequency_Hz[range(1,K+1)]
    slowness_spm = zeros(D);

    IM = eye(M);
    
    a_rd = aec.a_deg*pi/180
    e_rd = aec.e_deg*pi/180
    c_mps = aec.c_mps;
    
    slowness_spm = aec2theta(a_rd,e_rd,c_mps)
    delay_sec = dot(xsensors_m, slowness_spm);        

    dmu = array([-cos(a_rd)*cos(e_rd)/c_mps, sin(a_rd)*cos(e_rd)/c_mps,0])
    auxdmu = -2j*pi*dot(xsensors_m,dmu.reshape(3,1)) 
    FIM = 0
    for ik in range(K):
        D_ik = exp(-2j*pi*frequency_Hz[ik]*delay_sec);
        R_ik = dot(D_ik.reshape(M,1),D_ik.reshape(1,M).conjugate())+sigma2noise*IM
        invR_ik = inv(R_ik)
        d_ik = frequency_Hz[ik]* D_ik.reshape(M,1) * auxdmu.reshape(M,1);
        aux_drond_ik = dot(D_ik.reshape(M,1),d_ik.reshape(1,M).conjugate())
        drond_ik = aux_drond_ik+aux_drond_ik.transpose().conjugate()
        auxFIM = dot(invR_ik,drond_ik)
        FIM = FIM+(trace(dot(auxFIM,auxFIM)))
        
    CRBaz = real(1.0 / FIM)    
    return CRBaz