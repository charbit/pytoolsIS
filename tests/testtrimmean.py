# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:58:01 2016

@author: maurice
"""

import glob

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolutilities')


from numpy import nanmean,nanmedian, dot, array, nan, isnan
from numpy import random as rd
from matplotlib import pyplot as plt
from scipy.stats import trim_mean
from numpy import random
from tmc import trimmeancomplex

a     = 0.02#rd.rand()
N     = 100;
Nnan  = 15;
sqrtR = array([[1, 0.2],[0.04, 2]])
R     = dot(sqrtR,sqrtR.transpose())
xtab  = dot(R,rd.randn(2,N))
x     = xtab[0,:]+1j*xtab[1,:];
indperm = random.randint(1,N,N);
x[indperm[range(Nnan)]] = nan+1j*nan
mytt  = nanmean(trimmeancomplex(x,a,visu=1));
tt    = trim_mean(x,(1.0-a)/2.0)

from scipy import optimize as opt

def n1(u,x):
    indnotnan = 1-isnan(x)
    r=sum(abs(u[0]-x.real[indnotnan])+abs(u[1]-x.imag[indnotnan]))
    return r
    
init0 = [x[0].real, x[0].imag];
res = opt.fmin(n1,init0,args=(x,))
print res
print [nanmedian(x).real,nanmedian(x).imag]
print [nanmean(mytt).real, nanmean(mytt).imag]
#plt.plot(x.real,x.imag,'x')
#plt.hold(True)
#
#plt.plot(tt.real,tt.imag,'or')
#plt.plot(mytt.real, mytt.imag,'vg')
#plt.plot(res[0], res[1],'dm')
#plt.hold(False)






