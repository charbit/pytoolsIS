# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 07:55:28 2016

@author: maurice

#===========================
# This program usually sets a few parameters and
  calls the program 'estimazimvsCRB.py'
#===========================
"""


from numpy import linspace, zeros, arange, log10, random, array
from matplotlib import pyplot as plt
import time

#======================================
# run the following raw to know the station features
# execfile('liststations.py')
#======================================

listIS = ('I31',)
Lazmr = 1;
if Lazmr==1:
    listaz0_deg = array([84.0]) #array((random.randint(30),))+20.0
else:
    listaz0_deg  = arange(Lazmr)*360.0/Lazmr

llistSNR0_dB = 10
if llistSNR0_dB==1:
    listSNR0_dB = array([-5.0])
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
    print 'elapsed time %2.1f minuts' %((time.time()-t)/60.0)
    t=time.time()
    
#%%  
#plt.plot(listaz0_deg,20.0*log10(mzstdaz),'.-',label='MC')
#plt.hold('True')
##plt.plot(listaz0_deg,mzstdfromCRB,label='3D-CRB')
#plt.plot(listaz0_deg,20.0*log10(mzstdfromCRBazonly),label='az-only-CRB')
#plt.hold('False')
#plt.grid('on')
#plt.legend(loc='best');


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
