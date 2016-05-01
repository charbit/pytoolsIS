# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:25:32 2016

@author: maurice
"""

from numpy import array, log10, roots, linspace, zeros, max, sqrt
from matplotlib import pyplot as plt
# c =  snru2/(snru+1)(snru+rho)
# x2(1-1/c)+(1+rho)x+rho=0


c=0.98
L=10
listrho_dB = linspace(0,30.0,L)
snrU_dB = zeros(L)
rho = 10.0 ** (listrho_dB/10.0)
for il in range(L):
    d = (1.0+rho[il])**2-4.0*(1.0-1.0/c)*rho[il]
    bp = (-(1.0+rho[il])-sqrt(d))/2.0/(1.0-1.0/c)
    snrU_dB[il] = 10.0*log10(bp)

HorizontalSize = 5
VerticalSize   = 5
plt.close()   
figSNR = plt.figure(num=1,figsize=
               (HorizontalSize,VerticalSize), edgecolor='k', 
                facecolor = [1,1,0.92])

 
plt.plot(listrho_dB, snrU_dB,'.-')
plt.xlabel('R - dB')
plt.ylabel('SNR_u - dB')
plt.grid('on')
plt.show()