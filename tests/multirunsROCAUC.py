# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:50:22 2016

Compute the ROC curve associated to the FSTAT
for no LOC and LOC signals
Under H0 we have only noise
Under H1 we use the function "synthetizer"
     without and with LOC
     the true azimuth is randomly chosen
The maximization on Fstat is only w.r.t. the azimuth over
the range "range_azimuth_deg"

this program usually calls the program 
           "execfile('rocauc4noLOCLOC.py')"

@author: maurice
"""

from matplotlib import pyplot as plt
from numpy import array
from scipy.stats import f
from numpy import random

#Lruns0 = 200
#Lruns1 = 200
#flag_LOC = 1;

#=========================================
Lruns0 = 500
Lruns1 = 500
SOISNRrange_dB = -15.0*array([1.0, 1.0]);
SOIelevation0_deg = 70
SOIvelocity0_mps = 340
LOC_std_azimuth_deg = 5.0;
LOC_std_elevation_deg = 3.0;
LOC_std_velocity_mps = 13.0;

#=========================================
HorizontalSize = 5
VerticalSize   = 4
figrocauc = plt.figure(num=2,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
figrocauc.clf()

flag_LOC = 0;
execfile('rocauc4noLOCLOC.py')
a0 = a;
b0 = b;
eauc0 = eauc
std_eauc_boot0=std_eauc_boot

flag_LOC = 1;
execfile('rocauc4noLOCLOC.py')
a1 = a;
b1 = b;
eauc1 = eauc
std_eauc_boot1=std_eauc_boot

#%%
plt.plot(a0,b0,'.-b', label='without LOC')
plt.hold(True)
plt.plot(a1,b1,'.-r', label='with LOC')
plt.hold(False)

plt.axis('square')
plt.grid()
plt.legend(loc='best')
tt = 'without LOC: EAUC = %.2f (STD = %.2f)\nwitH LOC:  EAUC = %.2f (STD = %.2f)' % \
   (eauc0, std_eauc_boot0, eauc1, std_eauc_boot1)
plt.title(tt)

dirfigsave = '/Users/maurice/etudes/stephenA/propal2/figures'
hfile='%s/rocaucLOC%i.pdf' %(dirfigsave,flag_LOC)
#plt.savefig(hfile,format='pdf')





