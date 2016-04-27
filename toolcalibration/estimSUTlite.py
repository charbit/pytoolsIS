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
from numpy import real, imag, bool, arange

from scipy.linalg import sqrtm, eigh, svd
from scipy import fft, ifft , any, isreal, isnan, isinf
from numpy.linalg import matrix_rank, inv

from scipy.signal import lfilter, butter, cheby1, firwin
from scipy.signal import hamming, hanning

from matplotlib import pyplot as plt
from statsmodels.regression import yule_walker
from scipy.integrate import quad
from scipy.stats import norm

from scipy.stats import trim_mean

import time

from numpy import disp

#function [Rsup, freqslin, STDmoduleRlin, ...
#    STDphaseRlin_rd, nboverTHlin] = ...
#    estimSUTlite ...
#    (signals, structfiltercharacteristics, frequencylist_Hz, ...
#    Fs_Hz, MSCthreshold, trimpercent)
#========================================================================
# Synopsis:
# [Rsup, freqslin, STDmoduleRlin, ...
#     STDphaseRlin_rd, nboverTHlin] = ...
#     estimSUTlite ...
#     (signals, structfiltercharacteristics, frequencylist_Hz, ...
#     Fs_Hz, MSCthreshold, trimpercent)
#===============
# Inputs:
#     - signals : T x 2
#     - structfiltercharacteristics (FB structure)
#           see document
#     - frequencylist_Hz: array N x 1 of the selected frequencies
#       in Hz. N can take any value under Fs_Hz/2 with difference less
#       than around 5 or 6 times Fs_Hz/T
#     - Fs_Hz: sampling frequency in Hz
#     - MSCthreshold:
#     - trimpercent: percent of values keptfor averaging
#===============
# Outputs:
#     - Rsup: array N x 1 of the estimated ratios
#     - freqslin: array N x 1 of the selected frequencies
#       in Hz. almost the same as frequencylist_Hz, except if some 
#       are outside of the FB bandwidths.
#     - STDmoduleR: array N x 1 of the STD on the module of Rsup
#     - STDphaseR_rd: array N x 1 of the STD on the phase of Rsup
#     - nboverTH: array N x 1 of the number of values over the threshold
#========================================================================
def estimSUT(signals, structfiltercharacteristics, frequencylist_Hz,
           Fs_Hz, MSCthreshold, trimpercent):
    
    nbfrequencies          = len(frequencylist_Hz);
    Pfilter                = len(structfiltercharacteristics);
    nbfreqsbyfilter        = zeros(Pfilter);
    frequenciesinfilter_Hz = list(zeros(Pfilter));
    auxfreq                = zeros(1000);
    
    #=== determine the frequencies inside the bank filters
    #  in such a way that all frequencies are only in
    #  ONE filter band
    frequencylist_Hz_ii = frequencylist_Hz.copy();
    nbfrequencies_ii    = nbfrequencies;
    for idfilter in range(Pfilter):
        fc_if = structfiltercharacteristics[idfilter]
        fqlow_Hz    = fc_if.Wlow_Hz;
        fqhigh_Hz   = fc_if.Whigh_Hz;
        cp=0;
        for idf in range(nbfrequencies_ii):
            if (frequencylist_Hz_ii[idf]>fqlow_Hz) & (frequencylist_Hz_ii[idf]<=fqhigh_Hz):
                auxfreq[cp] = frequencylist_Hz_ii[idf]
                cp=cp+1;
        frequenciesinfilter_Hz[idfilter] = auxfreq[range(cp)];
        nbfreqsbyfilter[idfilter] = cp; 
        nbfreqsbyfilter[nbfreqsbyfilter==0] = nan;
    nbofallfrequencies = int(nansum(nbfreqsbyfilter));
    #========== we perform the filter coeffiecient from the structure
    #           denoted structfiltercharacteristics
    # using the Matlab functions as BUTTER.M
    filterbankcoeff = list(zeros(Pfilter));
    for idfilter in range(Pfilter):
        fc_if   = structfiltercharacteristics[idfilter];
        fname   = fc_if.designname;
        forder  = fc_if.Norder;
        fqlow   = fc_if.Wlow_Hz/Fs_Hz;
        fqhigh  = fc_if.Whigh_Hz/Fs_Hz;
        BW      = 2.0*array([fqlow,fqhigh])
        if fname == 'fir1':
            filnum = firwin(forder,fqlow,fqhigh)
            filden = 1;
        elif fname == 'butter':
                filnum,filden = butter(forder,BW, btype='pass');
        elif fname == 'cheby1':
                filnum,filden = cheby1(forder,0.02,BW, btype='pass');
                            
        filterbankcoeff[idfilter] = [filnum, filden]

    #========== we perform the shape window from the structure
    #           denoted structfiltercharacteristics
    # using the Matlab functions as HANN.M
    windshape = list(zeros(Pfilter));
    for idfilter in range(Pfilter):
        fc_if   = structfiltercharacteristics[idfilter];
        windowshapename = fc_if.windowshape;
        SCPperiod_sec   = fc_if.SCPperiod_sec;
        ratioDFT2SCP    = fc_if.ratioDFT2SCP;
        lengthDFT       = int(SCPperiod_sec*Fs_Hz/ratioDFT2SCP);
        if windowshapename == 'hann':
                windshape_ii = hanning(lengthDFT,sym=False);
                windshape[idfilter] = windshape_ii / sqrt(sum(windshape_ii ** 2));
    
    #==== pre-computation of the exponentials used by
    # the direct DFTs
    EXPV                = list(zeros(Pfilter));
    for idfilter in range(Pfilter):
        if not(isnan(nbfreqsbyfilter[idfilter])):
            nbfreq_idf      = int(nbfreqsbyfilter[idfilter]);
            fc_if           = structfiltercharacteristics[idfilter];
            SCPperiod_sec   = fc_if.SCPperiod_sec;
            ratioDFT2SCP    = fc_if.ratioDFT2SCP;
            lengthDFT       = int(SCPperiod_sec*Fs_Hz/ratioDFT2SCP);
            DFTindex        = array(range(lengthDFT))/float(Fs_Hz);
            DFTindex        = DFTindex.reshape(lengthDFT,1)
            freqaux         = frequenciesinfilter_Hz[idfilter].reshape(1,nbfreq_idf);
            EXPV[idfilter]  = exp(-2j*pi*dot(DFTindex,freqaux));
        else:
            EXPV[idfilter] = nan;
    
    #============================================
    #============================================

    Nsignals   = size(signals,0);
    R          = list(zeros(Pfilter));
    STDmoduleR = list(zeros(Pfilter));
    STDphaseR  = list(zeros(Pfilter));
    nboverTH   = list(zeros(Pfilter));
    MSC        = list(zeros(Pfilter));
    
    for idfilter in range(Pfilter):
        filnum = filterbankcoeff[idfilter][0];
        filden = filterbankcoeff[idfilter][1];
        filteredsignals = lfilter(filnum,filden,signals, axis=0);
        fc_if   = structfiltercharacteristics[idfilter];
        SCPperiod_sec   = fc_if.SCPperiod_sec;
        ratioDFT2SCP    = fc_if.ratioDFT2SCP;
        overlapDFT      = fc_if.overlapDFT;
        # Computation
        lengthDFT       = int(SCPperiod_sec*Fs_Hz/ratioDFT2SCP);
        lengthSCP       = int(SCPperiod_sec*Fs_Hz);
        DFTshift        = int((1-overlapDFT)*lengthDFT);
        NSCPwindows     = int(Nsignals/Fs_Hz/SCPperiod_sec);
        sigauxW         = zeros([lengthDFT,2]);
        
        if not(isnan(nbfreqsbyfilter[idfilter])):
            nbfreq_idf      = int(nbfreqsbyfilter[idfilter])
            
            SCP_ifreq11     = zeros([nbfreq_idf,NSCPwindows-1], dtype=complex);
            SCP_ifreq22     = zeros([nbfreq_idf,NSCPwindows-1], dtype=complex);
            SCP_ifreq12     = zeros([nbfreq_idf,NSCPwindows-1], dtype=complex);
            
            for iwindowSCP  in range(NSCPwindows-2):
                id0   = iwindowSCP*lengthSCP;
                id1   = 0;
                cpDFT = 0;
                while id1<id0+lengthSCP-lengthDFT:
                    id1    = id0 + cpDFT*DFTshift;
                    id2    = id1+lengthDFT;
                    
                    sigaux = filteredsignals[id1:id2,:];
                    sigauxW[:,0] = sigaux[:,0] * windshape[idfilter];
                    sigauxW[:,1] = sigaux[:,1] * windshape[idfilter];
                    for idfreq in range(nbfreq_idf):
                        X_ifreq1 = sum(sigauxW[:,0] * EXPV[idfilter][:,idfreq]);
                        X_ifreq2 = sum(sigauxW[:,1] * EXPV[idfilter][:,idfreq]);
                        SCP_ifreq11[idfreq,iwindowSCP] = SCP_ifreq11[idfreq,iwindowSCP]\
                                  + X_ifreq1 * conjugate(X_ifreq1);
                        SCP_ifreq22[idfreq,iwindowSCP] = SCP_ifreq22[idfreq,iwindowSCP]\
                                  + X_ifreq2 * conjugate(X_ifreq2);
                        SCP_ifreq12[idfreq,iwindowSCP] = SCP_ifreq12[idfreq,iwindowSCP] \
                                  + X_ifreq1 * conjugate(X_ifreq2);
                    cpDFT  = cpDFT+1;
           
            tabMSC_ifilter     = real(abs(SCP_ifreq12) ** 2) / real(SCP_ifreq11 * SCP_ifreq22);
             
            ind_ifilter_cst    = (tabMSC_ifilter>MSCthreshold);
            tabMSC_ifilter_cst = zeros([size(tabMSC_ifilter,0),size(tabMSC_ifilter,1)]);
            tabMSC_ifilter_cst.fill(nan)
            tabMSC_ifilter_cst[ind_ifilter_cst] = tabMSC_ifilter[ind_ifilter_cst];
            
            tabRsup_ifilter = SCP_ifreq11 / conjugate(SCP_ifreq12);       
                        
            tabRsup_ifilter_cst = zeros([size(tabRsup_ifilter,0),size(tabRsup_ifilter,1)],dtype=complex)
            tabRsup_ifilter_cst.fill(nan)
            
            tabRsup_ifilter_cst[ind_ifilter_cst] = tabRsup_ifilter[ind_ifilter_cst];
            
            tabRsup_ifilter_cst_trim = zeros([size(tabRsup_ifilter,0),size(tabRsup_ifilter,1)],dtype=complex)
            
            for ifre_idf in range(nbfreq_idf):
                tabaux = tabRsup_ifilter_cst[ifre_idf,:];
                tabRsup_ifilter_cst_trim[ifre_idf,:] = \
                       trimmeancomplex(tabaux,trimpercent);
            
            SCP_ifreq11_cst = zeros([size(SCP_ifreq11,0),size(SCP_ifreq11,1)],dtype=complex);
            SCP_ifreq11_cst.fill(nan)           
            SCP_ifreq11_cst[ind_ifilter_cst] = SCP_ifreq11[ind_ifilter_cst].copy();
            
            SCP_ifreq22_cst = zeros([size(SCP_ifreq22,0),size(SCP_ifreq22,1)],dtype=complex)
            SCP_ifreq22_cst.fill(nan)
            SCP_ifreq22_cst[ind_ifilter_cst] = SCP_ifreq22[ind_ifilter_cst].copy();
            
            tabR1122_cst = SCP_ifreq11_cst / SCP_ifreq22_cst;
                    
            weightMSCsupeta  = real(((tabMSC_ifilter_cst ** 2) \
                          /(1.0-tabMSC_ifilter_cst)) * tabR1122_cst);
            
            tabRsup_ifilter_cst_trimweightMSCsupeta = \
                        tabRsup_ifilter_cst_trim * weightMSCsupeta
            nansumden = weightMSCsupeta *(1-(isnan(tabRsup_ifilter_cst_trimweightMSCsupeta)));
                        
            R_filter = nansum(tabRsup_ifilter_cst_trimweightMSCsupeta,axis=1) \
                    / nansum(nansumden,axis=1);                               
            
            R[idfilter] = R_filter;
            
            nboverTH_ii = nansum(ind_ifilter_cst,axis=1);
            #===== perform STD on module and phase
            STDmoduleR[idfilter]   = nanstd(abs(tabRsup_ifilter_cst),axis=1) \
                            / sqrt(nboverTH_ii);
            STDphaseR[idfilter]    = nanstd(angle(tabRsup_ifilter_cst),axis=1) \
                            / sqrt(nboverTH_ii);
            nboverTH[idfilter]    = nboverTH_ii;
           
            MSC[idfilter] = tabMSC_ifilter;
            
        else:
            R[idfilter] = nan;
            STDmoduleR[idfilter] = nan;
            STDphaseR[idfilter]  = nan;
            nboverTH[idfilter] = nan
            MSC[idfilter] = nan;
        
    freqslin = zeros(nbofallfrequencies);
    Rsup     = zeros(nbofallfrequencies, dtype=complex);
    STDmoduleRlin = zeros(nbofallfrequencies);
    STDphaseRlin_rd = zeros(nbofallfrequencies);
    nboverTHlin = zeros(nbofallfrequencies);
    id2 = 0;
    for idfilter in range(Pfilter):
        if not(isnan(nbfreqsbyfilter[idfilter])):
            id1 = id2;
            id2 = id1+int(nbfreqsbyfilter[idfilter]);
            Rsup[id1:id2]     = R[idfilter].copy();
            freqslin[id1:id2] = frequenciesinfilter_Hz[idfilter];
            STDmoduleRlin[id1:id2] = STDmoduleR[idfilter];
            STDphaseRlin_rd[id1:id2] = STDphaseR[idfilter];
            nboverTHlin[id1:id2] =  nboverTH[idfilter];
        
    return Rsup, freqslin, STDmoduleRlin, STDphaseRlin_rd, nboverTHlin
#===================================================
#===================================================
#===================================================
#function trimmeanz = trimmeancomplex(z,trimpercent)
#%%
def trimmeancomplex(z,trimpercent):
# synospsis 
#     trim = trimmeancomplex(z,trimpercent)
# inputs: 
#         z: is an 1d array of length N of COMPLEX values
#         trimpercent: a value in (0,1)
#                if trimpercent=1, we have the trivial mean
# output:
#         trim: returns the trimmed mean of the complex values 
#================================================== 

    c = -2*log(1.0-trimpercent);
    N = len(z);
    Nnotnan = N-sum(isnan(z));
    trimz = zeros(size(z),dtype=complex)
   
    cp = 0;
    if Nnotnan > 20:
        zt = z.copy();
        zt[isnan(zt)] = 0.0;
        meanz = sum(zt)/Nnotnan;
        zc = zt-meanz * (1-(zt==0));
        zri      = zeros([2,N])
        zri[0,:] = real(zc);
        zri[1,:] = imag(zc);    
        R = dot(zri,transpose(zri))/(Nnotnan-1);
        
        Fm1 = inv(R);
        
        for ii in range(N):
            zri_ii = zri[:,ii];
            if dot(zri_ii.reshape(1,2), 
                   dot(Fm1, zri_ii.reshape(2,1))) < c \
                   and not(isnan(z[ii]).real)\
                   and not(isnan(z[ii]).imag):
                trimz[ii] = z[ii];
                cp=cp+1;
            else:
                trimz[ii] = nan + 1j*nan
    else:
        trimz = z.copy();        
    return trimz
    