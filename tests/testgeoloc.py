# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 07:55:55 2016

@author: maurice
"""


import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolutilities')

from geoloc import *

from matplotlib import  pyplot as plt
from numpy import array, ones, size, mean, zeros, dot, pi


xsensors_m =  1000*array([
    [-0.05997213,  0.194591122,   0.3911],
    [0.229169719,   0.083396195,   0.3921],
    [0.122158887,  -0.206822564,   0.3918],
    [-0.12375342,  -0.087843992,   0.3902],
    [-0.026664123,   0.015567290,   0.391],
    [0.919425013,   0.719431175,   0.3924],
    [0.183105453,  -1.103053672,   0.3831],
    [-1.2434694,   0.384734446,   0.3976]])

xsensors_m = xsensors_m-xsensors_m[0,:];

nbsensors = size(xsensors_m,0)

numselect = 'I31'


sensors = extractstationlocations(numselect, ReferenceEllipsoid=23)

xx=zeros([nbsensors,2])

for im in range(nbsensors):
    evi=sensors[im]
    xx[im,:]=[evi.geolocs.X_km,evi.geolocs.Y_km];

xx = xx - xx[0,:]

#plt.plot(xsensors_m[:,0],xsensors_m[:,1],'x')
plt.plot(xx[:,0]*1000.0,xx[:,1]*1000.0,'o')