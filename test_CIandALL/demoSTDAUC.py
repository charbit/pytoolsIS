# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:45:33 2016

@author: maurice
"""

#=============================================================
# IC_auc
# verification of the STD of the AUC
# 1) with asymptotic LCT and EXPON. model
# 2) with bootstrap approach
# 3) with Monte Carlo approach
# we choose a non gaussian distribution
#=========
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')

from geoloc import extractstationlocations

from toolIS import maxfstat, synthetizer
from toolIS import evalCRBwithgaussianLOC, CRBonazimuthonlywithoutLOC

from numpy import size, zeros, pi, cos, sin, int, intc, array, trace, nansum, max
from numpy import ceil, log2, exp, real, nan, std, log10, inf
from numpy import argmax, unravel_index, arange, histogram
from numpy import linspace,logspace, sum, mean, conj, concatenate
from numpy import dot, transpose, diag, sqrt, random, ones, eye, kron
from numpy.linalg import matrix_rank
from numpy import append

from scipy.linalg import sqrtm, eigh, svd, inv, pinv
from scipy import fft, ifft , any, isreal, isnan
from scipy.signal import hamming, hanning, resample, correlate
from scipy.signal import lfilter, butter
from scipy.integrate import quad
from scipy.stats import norm

from statsmodels.regression import yule_walker

from time import sleep
from matplotlib import pyplot as plt

from toolIS import asymPrecision, asymFalpha, asymRecall, rocauc

Lruns = 1;
stdauc_exp = zeros(Lruns);
stdauc_boot = zeros(Lruns);
N0 = 150;
N1 = 100;
MOY    = 1.0;
SIGMA1 = 2.0;
B      = 200;

for ir in range(Lruns):
    P0 = (random.randn(N0)) **2;
    P1 = (SIGMA1*random.randn(N1)+MOY) **2;
    
    alpha, beta, CIalpha, CIbeta, eauc, std_eauc_exp, \
               std_eauc_boot = rocauc(P0,P1,95,nbboot=B);
    stdauc_exp[ir]   = std_eauc_exp;
    stdauc_boot[ir]  = std_eauc_boot;

LrunsMC = 300;
aucsMC  = zeros(LrunsMC);
for ir in range(LrunsMC):
    P0 = (random.randn(N0)) ** 2;
    P1 = (SIGMA1*random.randn(N1)+MOY) **2;
    
    alpha, beta, CIalpha, CIbeta, eauc, std_eauc_exp, \
               std_eauc_boot = rocauc(P0,P1,95,nbboot=B);

    aucsMC[ir] = eauc;
    
stdauc_MC = std(aucsMC);
##
if Lruns>1:
    plt.plot(std_eauc_exp,'.-b')
    plt.hold('True')
    plt.plot(std_eauc_boot,'.-r')
    plt.plot((1, Lruns),stdauc_MC*ones(2),'.-k')
    plt.hold('False')
    plt.legend('Analytical expression','Bootstrap','Monte-Carlo')
    plt.grid()
    
    HorizontalSize = 12;
    VerticalSize   = 10;
else:
    print('MC std on %i runs: %5.4f\nEXP std : %5.4f\nBOOT std on %i resamples: %5.4f\n'% \
        (LrunsMC,stdauc_MC,std_eauc_exp,B,std_eauc_boot));

    
#%%
#===========
##
# verification of the values of B0 and B1
 # derived from exponential model agree with
 # those obtained by Monte-Carlo approach.
 #===========
N       = 10000;
P1_MC   = random.randn(N)+MOY;
PSI0_MC = norm.cdf(P1_MC,0,1);
B0_MC   = mean(PSI0_MC ** 2);

P0_MC   = random.randn(N);
PSI1_MC = (1.0-norm.cdf(P0_MC,MOY,1));
B1_MC   = mean(PSI1_MC ** 2);
A_MC    = mean(PSI1_MC);

alpha, beta, CIalpha, CIbeta, eauc, std_eauc_exp, \
    std_eauc_boot    = rocauc(P0,P1,95,nbboot=B);
eauc2   = eauc*eauc;
# theoretical expression
B0      = (2.0*eauc2) / (1.0+eauc);
B1      = eauc / (2.0-eauc);

print '%2.3f, %2.3f' %(B0, B0_MC)
print '%2.3f, %2.3f' %(B1, B1_MC)

#=============================================================

