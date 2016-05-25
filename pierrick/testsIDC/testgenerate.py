# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:51:58 2016

@author: maurice
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

import sys

mypaths = ('/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/pierrick/testsIDC/toolbox4IDC', )
Lmypaths = len(mypaths)

for ip in range(Lmypaths):
    if mypaths[ip] not in sys.path:
        sys.path.append(mypaths[ip])


from geoloc import extractstationlocations
from tool4generate import generator
from numpy import array, size, zeros,  isnan
from matplotlib import  pyplot as plt

#==========================================================================================
# This part extracts useful informatons about the sensors.
# station is a sequence of structures describing the sensosrs. It is computed from the
# geolocation of the sensors provided in file XLSfile. 
# That could be removed in the future 
# Here the only part we use is the sensor coordinates xsensors_m
numselect = 'I31'
XLSfile = '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/pierrick/testsIDC/allISs.xls';
station = extractstationlocations(XLSfile, numselect, ReferenceEllipsoid=23)
#==========================================================================================
nbsensors = len(station)
xsensors_m = zeros([nbsensors,3])
for im in range(nbsensors):
    evi   = station[im]
    xsensors_m[im,:] = array([evi.geolocs.X_km, \
                evi.geolocs.Y_km, \
                evi.geolocs.Z_km])*1000.0;
#==========================================================================================
#================ call the generator ======================================================
settingfile          = 'settings1.txt';
signals, groundtruth = generator(xsensors_m,settingfile)
#==========================================================================================
#==========================================================================================
#==========================================================================================
if isnan(signals.any()):
    print "too many detection demand"

else:
    SOIFs_Hz = 20
    M=size(xsensors_m,0)
    t_sec = array(range(size(signals,0)))/SOIFs_Hz
    t_hour = t_sec/3600.0
    tmps=array(range(size(signals,0)))/SOIFs_Hz;
    #=== plot
    HorizontalSize = 5
    VerticalSize   = 6
    figpdfsoi = plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
    plt.clf()
    for im in range(M):
        plt.subplot(M,1,im+1)
        plt.plot(tmps,signals[:,im])
        if im==M-1:
            plt.xlabel('time - s', fontsize=10)
            plt.xticks(fontsize=8)
            plt.yticks([])
        else:
            plt.xticks([])
            plt.yticks([])
            
    plt.show()


printfile = '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/pierrick/texte/'
# figpdfsoi.savefig('%sexample.png' %printfile)