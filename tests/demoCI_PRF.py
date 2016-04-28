# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:56:13 2016

@author: maurice
"""

# compare std of precision and recall
#
#
#
#

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

from toolIS import asymPrecision, asymFalpha, asymRecall

alpha_percent = 100*(1-2*(1-norm.cdf(1,0,1))) ; #68.27;
alpha_score = 0.5;

# p0 probability that Y=1 under H0 (cad false alarm)
p0        = 0.05;
N0        = 1000; 
m0        = N0*p0; # mean of FP)
s0        = N0*p0*(1-p0);
# p1 probability that Y=1 under H1 (cad true detection
p1        = 0.9;
N1        = 120;
m1        = N1*p1; # mean of TP
s1        = N1*p1*(1-p1);

bid, mean1Precisionth, sigma2Precisionth = asymPrecision(1.0,m0,m1,s0,s1);
sigmaPrecisionth = sqrt(sigma2Precisionth);

#[bid, mean1Fscoreth, sigma2Fscoreth] = asymFscore(1,N1,m0,m1,s0,s1);
#sigmaFscoreth    = sqrt(sigma2Fscoreth);

[bid, mean1Falphath, sigma2Falphath] = asymFalpha(1,alpha_score,N1,m0,m1,s0,s1);
sigmaFalphath    = sqrt(sigma2Falphath);

meanRecallth     = m1/N1;
sigmaRecallth    = sqrt(m1*(1.0-m1/N1)/N1/N1);

# CIexactth        = CIexactbin(1-p1,alpha_percent,N1);
# sigmaRecallth    = diff(CIexactth)/2;
# [sigmaRecallth-sigmaRecallappth]

Lruns     = 10000;
A         = zeros(Lruns);
Precision = zeros(Lruns);
Falpha    = zeros(Lruns);
Fscore    = zeros(Lruns);
Recall    = zeros(Lruns);
for ir in range(Lruns):
    Y0 = random.rand(N0,1)<p0;
    FP = float(sum(Y0));
    Y1 = random.rand(N1,1)<p1;
    TP = float(sum(Y1));
    Recall[ir] = TP/N1;
    A[ir] = FP/TP;
    Precision[ir] = 1.0 /(1.0+A[ir]);
#    Fscore[ir]    = 2.0 * Precision[ir]*Recall[ir]/(Precision[ir]+Recall[ir]);
    Falpha[ir]   = 1.0 / (alpha_score/Precision[ir] +(1.0-alpha_score)/Recall[ir]);



#===================
# Precision
npbinsP = 30;
ybinsP, xbinsP = histogram(Precision, bins=npbinsP, normed='True');
xbinsP = (xbinsP[1:npbinsP+1]+xbinsP[0:npbinsP])/2.0

lenzP = 300
zP    = mean1Precisionth+sigmaPrecisionth*linspace(-3.0,3.0,lenzP);
pzP = zeros(lenzP)
for izp in range(lenzP):
    zP_i = zP[izp]
    auxP = asymPrecision(zP_i,m0,m1,s0,s1);
    pzP[izp] = auxP[0]

#====================
# recall
npbinsR = 10;
ybinsR, xbinsR = histogram(Recall, bins=npbinsR, normed='True');
xbinsR = (xbinsR[1:npbinsR+1]+xbinsR[0:npbinsR])/2.0
lenzR = 300
zR = meanRecallth+sigmaRecallth*linspace(-3.0,3.0,lenzR);
pzR = zeros(lenzR)
for izr in range(lenzR):
    zR_i = zR[izr]
    auxR = asymRecall(zR_i,N1,m1,s1);
    pzR[izr] = auxR[0]
    
#====================
# Falpha
npbinsF = 30;
ybinsF,xbinsF = histogram(Falpha,npbinsF, normed='True');
xbinsF = (xbinsF[1:npbinsF+1]+xbinsF[0:npbinsF])/2.0
lenzF = 300
zF = mean1Falphath+sqrt(sigma2Falphath)*linspace(-3.0,3.0,lenzF);
pzF = zeros(lenzF)
for izr in range(lenzF):
    zF_i = zF[izr]
    auxF = asymFalpha(zF_i,alpha_score,N1,m0,m1,s0,s1);
    pzF[izr] = auxF[0]

    
#%%
HorizontalSize = 10
VerticalSize   = 3
figSTDasSNR = plt.figure(num=2,figsize=
               (HorizontalSize,VerticalSize), edgecolor='k', 
                facecolor = [1,1,0.92])

plt.subplot(1,3,1)
plt.plot(xbinsR,ybinsR,'o-')
plt.hold('True')
plt.plot(zR,pzR,'r')
plt.hold('False')
plt.title('Recall')

plt.subplot(1,3,2)
plt.plot(xbinsP,ybinsP,'o-')
plt.hold('True')
plt.plot(zP,pzP,'r')
plt.hold('False')
plt.title('Precision')

plt.subplot(1,3,3)
plt.plot(xbinsF,ybinsF,'o-')
plt.hold('True')
plt.plot(zF,pzF,'r')
plt.hold('False')
plt.title('Falpha')

plt.show()
#%%
#=== only ONE shot to compare the CI
#   performed by the function asympdfPrecision
#   and by the Monte-Carlo on Lruns runs.
Y0one = random.rand(N0)<p0;
FPone = 1.0*sum(Y0one);
Y1one = random.rand(N1)<p1;
TPone = 1.0*sum(Y1one);
FNone = 1.0*N1-TPone;
Aone  = FPone/TPone;
Rone  = TPone/N1;
Pone  = 1.0/(1.0+Aone);
Fone  = 2.0*Pone*Rone/(Pone+Rone);
Falphaone = 1.0 / (alpha_score/Pone +(1.0-alpha_score)/Rone);
 
m0chap = 1.0*sum(Y0one);
m1chap = 1.0*sum(Y1one);
s0chap = m0chap*(1.0-m0chap/N0);
s1chap = m1chap*(1.0-m1chap/N1);

bid, hatmean1Precision, hatsigma2Precision = \
        asymPrecision(Pone,m0chap,m1chap,s0chap,s1chap)
hatsigmaPrecision = sqrt(hatsigma2Precision);

bid, hatmean1Falpha, hatsigma2Falpha = \
    asymFalpha(Falphaone,alpha_score,N1,m0chap,m1chap,s0chap,s1chap);
sigmaFalpha = sqrt(hatsigma2Falpha);
sigmaRecall = sqrt(m1chap*(1.0-m1chap/N1)/N1/N1);


# for N0 and N1 high, the approx CI is OK. 
# CIestim        = CIexactbin(1-m1chap/N1,alpha_percent,N1);
# sigmaRecall    = diff(CIestim)/2;

#===
print('%s & %5.4f &%5.4f &%5.4f \\' %('$precision$', sigmaPrecisionth, hatsigmaPrecision, std(Precision)));
print('%s & %5.4f &%5.4f &%5.4f \\' %('$recall$', sigmaRecallth, sigmaRecall, std(Recall)));
print('%s & %5.4f &%5.4f &%5.4f \\' %('$alpha$', sigmaFalphath, sigmaFalpha, std(Falpha)));


#save -ascii ../../finalreport/statPRFwithIC.txt  txtF txtP txtR txtFa
# figure(1); print -depsc -loose ../../finalreport/figures/statPRFwithIC.eps

