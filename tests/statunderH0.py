# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 18:17:32 2016

@author: maurice

#================================= 
Synopsis:
This program performs the statistic under H0 of the
following functions of test:
    - Fisher
    - Fstat
It also computes the true distribution
    - for Fisher, it is a F with N,N(M-1) dof
    - for Fstat, we have no closed form
      expression but we use Monte-Carlo
      approach.
#================================= 
"""

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
         
import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/pytools/progspy/toolutilities')

from geoloc import extractstationlocations
from toolIS import evalCRBwithgaussianLOC,maxfstat, UpsilonXi, geneFZ, geneFF
from toolIS import pvalunderH0
from numpy import array,ones, exp, size, pi, arange, sort
from numpy import zeros, dot, diag, cos, sin, sqrt, linspace

from numpy import random

from scipy.stats import f, norm

from matplotlib import pyplot as plt
    
gridaz_deg = array(linspace(0,360,1))
gridel_deg = array(linspace(0,90,1))
gridc_mps = array(linspace(300,400,1))

Fs_Hz = 20

station = 'I31'

sensors = extractstationlocations(station, ReferenceEllipsoid=23)
nbsensors = len(sensors)
xsensors_m = zeros([nbsensors,3])
if nbsensors == 0:
    sys.exit('no sensor in %s'%station)
else:
    for im in range(nbsensors):
        evi   = sensors[im]
        xsensors_m[im,:] = array([evi.geolocs.X_km,
                  evi.geolocs.Y_km, 
                  evi.geolocs.Z_km])*1000.0;
    
    xsensors_m = xsensors_m - xsensors_m[0,:]
    M = size(xsensors_m,0);
T_sec = 30 ; 
N = int(T_sec*Fs_Hz)
Lruns = 10000
Fstat = zeros(Lruns)

nbH0                = 500;
La                  = 25;
Le                  = 2;
Lc                  = 2;
Lruns_CLT           = 10000;
if 0:
    range_azimuth_deg   = sort(random.rand(La)*180.0) #linspace(0,180,La);   # 
    range_elevation_deg = sort(random.rand(Le)*90.0 )# linspace(30,80,Le);   #
    range_velocity_mps  = sort(300.0+random.rand(Lc)*40.0) #linspace(300,400,Lc); # 
else:
    range_azimuth_deg   = linspace(0,180,La);   # 
    range_elevation_deg = linspace(30,50,Le);   #
    range_velocity_mps  = linspace(300,340,Lc); # 
# size of the dictionary
#
Q                   = La*Le*Lc
M2                  = M*M;
twoQ                = 2*Q;

Upsilon,Xi = UpsilonXi(xsensors_m, Fs_Hz, range_azimuth_deg, 
                  range_elevation_deg, range_velocity_mps)
                 
Xicorr         = zeros([Q,Q])
for i in range(Q):
    for j in range(Q):
        Xicorr[i,j] = Xi[i,j] /sqrt(Xi[i,i]*Xi[j,j])
        
cQ             = Q*(Q+1)/2
Xivalues       = zeros(cQ)
cp             = 0
for i in range(Q):
    for j in range(i,Q):
        Xivalues[cp]=Xicorr[i,j]
        cp=cp+1

print sum(Xivalues==1), cQ

#%
#HorizontalSize = 6
#VerticalSize   = 4
#
#fighistmat = plt.figure(num=8,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92])
#hbinsXivalues, binsXivalues, patches = plt.hist(Xivalues, bins=30, normed=0, facecolor='g', alpha=0.2)
#
#plt.plot(binsXivalues[0:30]+(binsXivalues[1]-binsXivalues[0])/2, hbinsXivalues,'o')
#plt.grid()

#if Q==1:
#    plt.savefig(dirsavefigures + 'corrXidistributionclosevalues.pdf')
#else:
#    plt.savefig(dirsavefigures + 'corrXidistributionfarvalues.pdf')


                  
#%%
#================== SIMULATION ========================
# simulation with Lruns runs
# under H0, the signa is only noise
maxFsimul    = zeros([nbH0,4])
for ir in range(nbH0):
    if (ir-10*(ir/10)) == 0: print ir
    x             = random.randn(N,M)
    maxFsimul[ir] = maxfstat(x, Fs_Hz, xsensors_m, range_azimuth_deg,
        range_elevation_deg,range_velocity_mps);
#%
maxFZ     = geneFZ(Lruns_CLT, N, xsensors_m, Fs_Hz, range_azimuth_deg, 
                   range_elevation_deg, range_velocity_mps)
maxFF     = geneFF(Lruns_CLT, N, xsensors_m, Fs_Hz, range_azimuth_deg, 
                  range_elevation_deg, range_velocity_mps)
FF        = maxFsimul[0:ir+1,0]

#== compute the p-value with the asymptotic distribution
#   (not independent)
ppv  = pvalunderH0(FF, N, xsensors_m, Fs_Hz,
                   range_azimuth_deg, 
                   range_elevation_deg, range_velocity_mps);

# pvalues with he limG independent and Findependent
ppvG = 1-norm.cdf(FF,1.0,sqrt(2.0*M/(M-1.0)/N))**Q;
ppvF = 1-f.cdf(FF,N,N*(M-1))**Q;

# pdf of the max of the limG independent and Findependent
linx        = linspace(0.69,1.3,200)
sigmaGlim   = sqrt(2.0*M/(M-1.0)/N)
nu1         = N
nu2         = N*(M-1)
pdffromF    = f.pdf(linx,nu1,nu2)
pdffromFind = Q * pdffromF * (f.cdf(linx,nu1,nu2)**(Q-1));
pdffromGind = Q * norm.pdf(linx,1.0,sigmaGlim) * (norm.cdf(linx,1.0,sigmaGlim)**(Q-1));


dirfigsave = '/Users/maurice/etudes/stephenA/propal2/figures/'

#%%
#
#HorizontalSize = 6
#VerticalSize   = 6
#figpvalFoT     = plt.figure(num=1,figsize=(HorizontalSize,VerticalSize), 
#                            edgecolor='k', facecolor = [1,1,0.92]);
#plt.subplot(2,1,1)
#plt.ylabel("Frequency")
#plt.title("based on the asymptotic distribution")
#n, bins, patches = plt.hist(ppv, bins=nbbins, normed=1, facecolor='g', alpha=0.2);
#plt.xlim(0,1)
#
#plt.subplot(2,1,2)
#plt.ylabel("Frequency")
#plt.title("$F_{N,N(M-1)}$ distribution")
#n, bins, patches = plt.hist(ppvF, bins=nbbins, normed=1, facecolor='g', alpha=0.2);
##plt.xlim(0,1)
#
#if 1:    
#    figpvalFoT.savefig(dirsavefigures + 'distribFoTunderH0.pdf')
##
#nbbins = 10;
##%% not used
#HorizontalSize = 6
#VerticalSize   = 6
#figpvalFoTappr = plt.figure(num=2,figsize=(HorizontalSize,VerticalSize), 
#                            edgecolor='k', facecolor = [1,1,0.92]);
#plt.subplot(2,1,1)
#plt.ylabel("Frequency")
#plt.title("based on the asymptotic distribution")
#n, bins, patches = plt.hist(ppv, bins=nbbins, normed=1, facecolor='g', alpha=0.2);
#plt.xlim(0,1)
#
#plt.subplot(2,1,2)
#plt.ylabel("Frequency")
#plt.title("$\mathcal{N}(1,2M/N(M-1))$ distribution")
#n, bins, patches = plt.hist(ppvG, bins=nbbins, normed=1, facecolor='g', alpha=0.2);
##plt.xlim(0,1)
#
#if 0:    
#    figpvalFoTappr.savefig(dirsavefigures + 'ppvwithouraglovsfisher.pdf')



#%%
##================= compute the histograms
HorizontalSize = 10
VerticalSize   = 4

figsimul=plt.figure(num=2,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
plt.subplot(1,2,1)
n, bins, patches = plt.hist(FF, bins=10, normed=1, facecolor='b')
plt.plot(linx,pdffromF, color='r')
plt.xlim(0.8,1.3)
plt.ylim(0,15)
plt.title("Direct simulation")  

plt.subplot(1,2,2)
n, bins, patches = plt.hist(maxFF, bins=10, normed=1, facecolor='b')
#plt.plot(linx,pdffromFind, color='r')
plt.ylabel("Frequency")
plt.xlim(0.8,1.3)
plt.ylim(0,15)
plt.title("Asymptotic distribution")    
   
#plt.subplot(3,1,3)
#n, bins, patches = plt.hist(maxFZ, bins=10, normed=1, facecolor='g', alpha=0.2)
#plt.plot(linx,pdffromGind, color='r')
#xnorm = linspace(0,3,200)
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.xlim(0.9,1.3)
#plt.ylim(0,20)
#plt.title("based on asmptotic distribution - II")    
#  
if Q==1:
    figsimul.savefig(dirfigsave + 'texte/simul0.pdf')
elif La==100:
    figsimul.savefig(dirfigsave + 'simul1.pdf')
elif La==25:
    figsimul.savefig(dirfigsave + 'simul2.pdf')

#%%
#figmattriceXi=plt.figure(num=2,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92])
#plt.imshow((Xi))
#figmattriceXi.savefig(dirsavefigures + 'matrixXi.pdf')
##%% not used
#HorizontalSize = 6
#VerticalSize   = 6
#
#cQ             = Q*(Q+1)/2
#Xivalues       = zeros(cQ)
#cp             = 0
#for i in range(Q-1):
#    for j in range(i,Q):
#        Xivalues[cp]=Xi[i,j]
#        cp=cp+1
#
#print sum(Xivalues==0)
#
#fighistmat = plt.figure(num=8,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92])
#n, bins, patches = plt.hist(Xivalues, bins=30, normed=1, facecolor='g', alpha=0.2)
#fighistmat.savefig(dirsavefigures + 'fighistmat.pdf')


##%% not used
#HorizontalSize = 6
#VerticalSize   = 6
#figpvalFappronly = plt.figure(num=3,figsize=(HorizontalSize,VerticalSize), 
#                            edgecolor='k', facecolor = [1,1,0.92]);
#
#plt.ylabel("Frequency")
#plt.title("$\mathcal{N}(1,2M/N(M-1))$ distribution")
#plt.subplot(2,1,1)
#n, bins, patches = plt.hist(ppvG, bins=10, normed=1, facecolor='r', alpha=0.2);
#plt.subplot(2,1,2)
#n, bins, patches = plt.hist(ppvF, bins=10, normed=1, facecolor='g', alpha=0.2);
##plt.xlim(0,1)
#
#if 0:    
#    figpvalFappronly.savefig(dirsavefigures + 'approxppv.pdf')
#

#%% not used
#nbbins = 20
#HorizontalSize = 6
#VerticalSize   = 6
#
#figpdfQ=plt.figure(num=4,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
#
#
#n, bins, patches = plt.hist(FF, bins=nbbins, normed=1, facecolor='g', alpha=0.2)
#n, bins, patches = plt.hist(maxFF, bins=nbbins, normed=1, facecolor='r', alpha=0.2)
#
##plt.plot(linx,pdffromGind, color='r')
#plt.plot(linx,pdffromF, color='b')
#plt.plot(linx,pdffromFind, color='r')
#
#if 0:    
#    figpdfQ.savefig(dirsavefigures + 'approxpdfQ.pdf')

#%% not used
#pdfchi2        = chi2.pdf(linx*N,nu1)
#HorizontalSize = 6
#VerticalSize   = 6
#
#figpdfchi2 = plt.figure(num=5,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
#
#n, bins, patches = plt.hist(M*log(1.0+FF/(M-1)), bins=nbbins, normed=1, facecolor='g', alpha=0.2)
#
#plt.plot(linx,pdfchi2*N, color='b')
