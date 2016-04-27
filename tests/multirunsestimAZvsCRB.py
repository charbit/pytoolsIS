# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 07:55:28 2016

@author: maurice

#===========================
# This program launches the program 'estimazimvsCRB.py'
#===========================
"""
#listIS = ('I02','I04','I05','I06','I07','I08','I09',
#     'I10','I11','I13','I14','I17','I18','I19','I21',
#     'I23','I24','I26','I27','I30','I31','I32','I33',
#     'I34','I35','I36','I37','I39','I40','I41','I42',
#     'I43','I44','I45','I46','I47','I48','I49','I50',
#     'I51','I52','I53','I55','I56','I57','I58','I59')

#listIS = ('I27','I30','I31','I32','I33',
#     'I34','I35','I36','I37','I39','I40','I41','I42',
#     'I43','I44','I45','I46','I47','I48','I49','I50',
#     'I51','I52','I53','I55','I56','I57','I58','I59')
#
#02:M=5
#22:M=4
#27:M=18
#30:M=6
#31:M=8
#32:M=7
#33:M=8 (with a few co-located)
#34,M=12 (with a few co-located)
#45:M=4

from numpy import linspace, zeros, arange, log10, random, array
from matplotlib import pyplot as plt
import time

listIS = ('I22',)
Lazmr = 12;
if Lazmr==1:
    listaz0_deg = array((random.randint(360),))
else:
    listaz0_deg  = arange(Lazmr)*360.0/Lazmr

llistSNR0_dB = 1
if llistSNR0_dB==1:
    listSNR0_dB = linspace(-5.0,5.0,llistSNR0_dB)
else:
    listSNR0_dB = linspace(-25.0,25.0,llistSNR0_dB)
    
mzstdaz = zeros([Lazmr,llistSNR0_dB])
mzstdfromCRB = zeros([Lazmr,llistSNR0_dB])
mzstdfromCRBazonly = zeros([Lazmr,llistSNR0_dB])

t=time.time()

for ia in range(Lazmr):
    az0_deg = listaz0_deg[ia]
    execfile('estimazimvsCRB.py')
    mzstdaz[ia,:]=stdazMC_deg
    mzstdfromCRB[ia,:]=stdazCRB_deg
    mzstdfromCRBazonly[ia,:]=stdazonly_CRB_deg
    print 'elapsed time %2.1f%time.time()-t
    t=time.time()
    
#%%  
plt.plot(listaz0_deg,20.0*log10(mzstdaz),'.-',label='MC')
plt.hold('True')
#plt.plot(listaz0_deg,mzstdfromCRB,label='3D-CRB')
plt.plot(listaz0_deg,20.0*log10(mzstdfromCRBazonly),label='az-only-CRB')
plt.hold('False')
plt.grid('on')
plt.legend(loc='best');


#%%
dirsavefigures = '/Users/maurice/etudes/stephenA/propal2/figures/'
HorizontalSize = 5
VerticalSize   = 3
figSTDasSNR = plt.figure(num=2,figsize=
               (HorizontalSize,VerticalSize), edgecolor='k', 
                facecolor = [1,1,0.92])
                
plt.plot(listSNR0_dB, 20.0*log10(mzstdaz[0,:]),'.-b',label='MC')
plt.hold('True')
#plt.plot(listSNR0_dB, 20*log10(mzstdfromCRB[0,:]),'g',label='CRB')
plt.plot(listSNR0_dB, 20*log10(mzstdfromCRBazonly[0,:]),'.-r',label='CRB')
plt.hold('False')
plt.legend()
plt.grid()
plt.xlabel('SNR - dB')
plt.ylabel('STD - dB')
plt.show()
if 0:
    dirsavefigures = '/Users/maurice/etudes/stephenA/propal2/figures/'
    figSTDasSNR.savefig(dirsavefigures + \
      'STDasSNRLOC%i%s.pdf'%(flag_LOC,station))
