# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:25:32 2016

performs the SNR on one channel, assuming 
the MSC has a given value and the ratio 
between the SNB on the 2 channels is known
             1
  MSC = ------------------------
        (1+SNR_u^-1)(1+SNR_r^-1)
with SNR_u = rho SNR_r
        
@author: maurice
"""

from numpy import array, log10, roots, linspace, zeros, max, sqrt
from matplotlib import pyplot as plt
# c =  snru2/(snru+1)(snru+rho)
# x2(1-1/c)+(1+rho)x+rho=0

HorizontalSize = 4
VerticalSize   = 4
plt.close()   
figSNR = plt.figure(num=1,figsize=
               (HorizontalSize,VerticalSize), edgecolor='k', 
                facecolor = [1,1,0.92])
    
cp=1
for c in linspace(0.90,0.99,4):
    L=10
    listrho_dB = linspace(0,30.0,L)
    snrU_dB = zeros(L)
    rho = 10.0 ** (listrho_dB/10.0)
    for il in range(L):
        d = (1.0+rho[il])**2-4.0*(1.0-1.0/c)*rho[il]
        bp = (-(1.0+rho[il])-sqrt(d))/2.0/(1.0-1.0/c)
        snrU_dB[il] = 10.0*log10(bp)
    plt.plot(listrho_dB, snrU_dB,'.-',label='MSC = %2.2f'%c)
    plt.hold('True')
    plt.xlabel('R - dB' , family = 'times',fontsize=8 )
    plt.ylabel('SNR_u - dB', family = 'times',fontsize=8)
    plt.grid('on')
    plt.show()
    cp=cp+1

plt.hold('False')
plt.legend(fontsize=8, loc='best')
plt.xticks( family = 'times',fontsize=8 )
plt.yticks( family = 'times',fontsize=8 )
