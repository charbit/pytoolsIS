# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# test of the function asymFalpha,
# asymptotic distribution of the Falphascore



import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')

from geoloc import extractstationlocations
from toolIS import evalCRBwithgaussianLOC,maxfstat, UpsilonXi, geneFZ, geneFF
from toolIS import pvalunderH0, asymptoticFalpha


from numpy import mean,std, zeros
from scipy import random


# statistic of the false positives (FP)
p0=0.02
N0=3000
# LCT of the FP sum
m0=N0*p0
sigma20=p0*(1-p0)*N0

# statistic of the true positives
p1=0.9
N1=200
# LCT of the TP sum
m1=N1*p1
sigma21=p1*(1-p1)*N1

alpha=0.2
Lruns = 3000
F=zeros(Lruns)
for ir in range(Lruns):
    X0=random.rand(N0,1)<p0
    X1=random.rand(N1,1)<p1
    FP=float(sum(X0))
    TP=float(sum(X1))
#    FN=N1-TP
#    recall=TP / (TP+FN)
    recall = TP/N1
    precision = TP/(TP+FP)
    Fm1=alpha/precision+(1-alpha)/recall
    F[ir]=1.0/Fm1

meanpz, varpz, pz = asymptoticFalpha(alpha,N1,m0,m1,sigma20,sigma21,x=0.1)

print [meanpz, mean(F)]

print [varpz**0.5, std(F)]