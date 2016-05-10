# -*- coding: utf-8 -*-
"""
Created on Fri May  6 06:54:40 2016

@author: maurice
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)


import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')
from matplotlib import pyplot as plt


from numpy import zeros, dot, ones, mean, pi, sort, arange, log2
from toolIS import CRBonazimuthonlywithoutLOC, rotate2D, stationcharacteristics
from numpy import random, array, exp, sqrt, argmax

M=8
Lruns = 5000;
distrib_x=zeros(Lruns)
distrib_y=zeros(Lruns)

rangex0 = 1.0;#*1.7;
rangex1 = 1.0;#*0.75;
maxdmax = sqrt(rangex0**2+rangex1**2)
maxentropy = log2(M*(M-1)/2)
x=zeros([M,2,Lruns])

for ir in range(Lruns):

        x[:,0,ir] = rangex0*random.rand(M)-rangex0/2.0;
        x[:,1,ir] = rangex1*random.rand(M)-rangex1/2.0;
        
        xcht = stationcharacteristics(x[:,:,ir])
        R2 = xcht[2]
        dmax = xcht[3]
        dmin = xcht[4]
        entropy_d = xcht[8]
        entropy_o = xcht[9]
        dispersionratio_d = xcht[10]
        dispersionratio_o = xcht[11]
        
        distrib_x[ir] = dmin;
        distrib_y[ir] = entropy_d

#%%
plt.clf()
indmax=argmax(distrib_x);
xwithmin = x[:,:,indmax];
plt.plot(xwithmin[:,0],xwithmin[:,1],'o')
plt.xlim([-0.5,0.5])
plt.ylim([-0.5,0.5])
plt.axis('square')

print [distrib_x[indmax]/maxdmax, distrib_y[indmax]/maxentropy]