# attention the signal vectors and its Fourier transforms
# are ROW vetors.

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolseismic')

import estimdirections
from estimdirections import matrixtrihedron, extract1direction, extract1directionSCP

from numpy import zeros, array, size, real, ones, arctan2, arcsin
from numpy import pi, dot, mean, std
import scipy.io as io
from scipy.signal import lfilter, butter

from matplotlib import pyplot as plt

# the matlab file contains the variable 'signals_centered'
# 'io.loadmat' can read matlab file
matfile = io.loadmat('../toolseismic/year2015month10day22.mat')
signals = matfile['signals_centered']


Fs_Hz   = 40;
BW_Hz   = 10.0;
num,den = butter(2,2.0*BW_Hz/Fs_Hz, btype='low');

# theoretically the Oz azimut has no effect 
# because the Oz elevation is 90

azimutREF_deg    = array([0.0, 90.0, 0.0]);
elevationREF_deg = array([0.0, 0.0, 90.0]);

VREF             = matrixtrihedron(azimutREF_deg, elevationREF_deg);
grammianVREF     = dot(VREF,VREF.transpose());

#%%
#==================================

window_sec     = 300;
T              = size(signals,0);
N              = int(Fs_Hz*window_sec);
Nshift         = int(N*0.5);
nb_windows     = int(T/Nshift);nb_windows=1
Huf            = ones([N,3]);
Hrf            = ones([N,3]);

signals_filtered = zeros([size(signals,0),size(signals,1)]);
for ik in range(6):
    signals_filtered[:,ik] = lfilter(num,den,signals[:,ik]);

hatazimutSUT_deg       = zeros([3,nb_windows]);
hatelevationSUT_deg    = zeros([3,nb_windows]);
for ir in range(nb_windows):
    id1 = ir*Nshift;
    id2 = min([id1+N,T]);   
    N_ir = id2-id1;
    xutnoise = zeros([N_ir,3]);
    xrtnoise = zeros([N_ir,3]);
    xutnoise[0:N_ir,0] = signals_filtered[id1:id2,0];
    xutnoise[0:N_ir,1] = signals_filtered[id1:id2,1];
    xutnoise[0:N_ir,2] = signals_filtered[id1:id2,2];
    xrtnoise[0:N_ir,0] = signals_filtered[id1:id2,3];
    xrtnoise[0:N_ir,1] = signals_filtered[id1:id2,4];
    xrtnoise[0:N_ir,2] = signals_filtered[id1:id2,5];
    
    #== start computation
    for kk in range(3):
        voptim = extract1direction(\
            xutnoise[:,kk], xrtnoise, Huf[0:N_ir,kk], Hrf[0:N_ir,:], VREF);
        hatazimutSUT_deg[kk,ir] = real(arctan2(voptim[1],voptim[0]))*180.0/pi;
        hatelevationSUT_deg[kk,ir] = real(arcsin(voptim[2]))*180.0/pi;

##
print hatazimutSUT_deg
#%%
#HorizontalSize = 12
#VerticalSize   = 6
#fighist        = plt.figure(num=2,figsize=(HorizontalSize,VerticalSize), 
#                            edgecolor='k', facecolor = [1,1,0.92]);
#nbbins = 20;
#for kk in range(3):
#    plt.subplot(2,3,kk+1)
#    n, bins, patches = plt.hist(hatazimutSUT_deg[kk,:], bins=nbbins, normed=0, facecolor='g', alpha=0.2);
#    plt.subplot(2,3,kk+4)
#    n, bins, patches = plt.hist(hatelevationSUT_deg[kk,:], bins=nbbins, normed=0, facecolor='g', alpha=0.2);
#
##%%
#meanazSUT = mean(hatazimutSUT_deg,axis=1);
##stdazSUT  = std(hatazimutSUT_deg,[],2);
##
#meanelSUT = mean(hatelevationSUT_deg,axis=1);
##stdelSUT  = std(hatelevationSUT_deg,[],2);
##
#hatVSUT = matrixtrihedron(meanazSUT,meanelSUT);
##
#grammianVSUT = dot(hatVSUT,hatVSUT.transpose());
##
#print meanazSUT
#print meanelSUT
#
#print grammianVSUT
#print grammianVREF
