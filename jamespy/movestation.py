# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:39:40 2016

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
from numpy import random, array, exp


#center = [2,6];
#x = random.rand(8,2)
#y = rotate2D(x,center,30)
#plt.plot(x[:,0],x[:,1],'o')
#plt.hold(True)
#plt.plot(y[:,0],y[:,1],'or')
#plt.plot(center[0],center[1],'ok')
#plt.hold(False)
#plt.grid('on')

rangex0 = 1 #.7;
rangex1 = 1#0.75;
maxmax = sqrt(rangex0**2+rangex1**2)

alpha_deg = 45;
center=[10,20]
M=8;

x = zeros([M,2])
combi=M*(M-1)/2;
slope =1;
oneshot=1;
entropy_d=0
max_entropy = log2(combi)
dispersionratio_d = 100;
dispersionratio_o = 100;
cpt=0
dmin = 0

while ((dispersionratio_d>1.0) | (entropy_d/max_entropy <0.9) | (dmin<0.2*maxmax)) & (oneshot) :  
    if 1:
        x=zeros([M,2])
        x[:,0] = rangex0*random.rand(M)-rangex0/2.0;
        x[:,1] = rangex1*random.rand(M)-rangex1/2.0;
        cpt=cpt+1

    else:
        oneshot = 0;
        
        rho = exp(2j*pi*arange(M)/M);
        x[:,0] = rho.real*0.8
        x[:,1] = rho.imag/3.0; 
        
#        x = array([(-0.5, 0.90625),
#                   (0.9609375, 1.390625),
#             (1.453125, -0.0703125),
#             (0.5, -1.5),
#             (-0.421875, -0.484375),
#             (-0.2890625, 0.0078125),
#             (-0.0546875, 0.25),
#             (-0.109375, -0.1875)])
#        x=x/2.0
        
    xc = x-dot(ones(M).reshape(M,1),mean(x,0).reshape(1,2))
    xcht = stationcharacteristics(x)
    slope = xcht[2]
    dmin = xcht[4]
    entropy_d = xcht[8]
    entropy_o = xcht[9]
    dispersionratio_d = xcht[10]
    dispersionratio_o = xcht[11]
    
    distance = xcht[6]
    orient = xcht[7]
    
#    print [slope, entropy_d, entropy_o]
   
    
y=rotate2D(x,center,alpha_deg)
yc=y-dot(ones(M).reshape(M,1),mean(y,0).reshape(1,2)) #rotate2D(xc,center,alpha_deg)

x_m = zeros([M,3])
x_m[:,0:2] = 1000.0*x;
x_m[:,2] = 30.0*random.rand(M);

xc_m = zeros([M,3])
xc_m[:,0:2] = 1000.0*xc;
xc_m[:,2] = x_m[:,2]

y_m = zeros([M,3])
y_m[:,0:2] = 1000.0*y;
y_m[:,2] = x_m[:,2]

yc_m = zeros([M,3])
yc_m[:,0:2] = 1000.0*yc;
yc_m[:,2] = x_m[:,2]

xccht= stationcharacteristics(xc)
ycht= stationcharacteristics(y)
yccht= stationcharacteristics(yc)

dmin = xcht[4]
dmax = xcht[5]


#print 'R = %5.2f, area = %5.2f, Rc = %5.2f, areac = %5.2f' %(xcht[0], xcht[1], xccht[0], xccht[1])
#print 'R = %5.2f, area = %5.2f, Rc = %5.2f, areac = %5.2f' %(ycht[0], ycht[1], yccht[0], yccht[1])
#print 'R2_d = %5.4f, slope = %5.4f, slope = %5.4f, slope = %5.4f'%(xcht[2], xccht[2], ycht[2], yccht[2])



print '*********************'
print 'R2_d = %5.4f, entropyRatio_d = %5.2f, entropyRatio_o = %5.2f, dispersion-d = %5.2f' %(xcht[2], entropy_d/max_entropy, entropy_o/max_entropy, dispersionratio_d)
print '*********************'

sigma2noise = 100.0;
T_sec = 10.0;
Fs_Hz = 20.0;
aec = struct(a_deg=30.0,e_deg=45.0,c_mps=340);

#print CRBonazimuthonlywithoutLOC(xc_m, sigma2noise, aec,T_sec , Fs_Hz);
#print CRBonazimuthonlywithoutLOC(y_m, sigma2noise, aec,T_sec , Fs__init
#print CRBonazimuthonlywithoutLOC(yc_m, sigma2noise, aec,T_sec , Fs_Hz);


#%%
mxu = 1.0;

plt.clf()
plt.subplot(211)
plt.plot(x[:,0],x[:,1],'o')
plt.hold(True)
plt.plot(y[:,0],y[:,1],'o')
plt.plot([-rangex0/2.0,rangex0/2.0], [rangex1/2.0,rangex1/2.0],linewidth=2,color='k',linestyle='--')
plt.plot([rangex0/2.0,rangex0/2.0], [rangex1/2.0,-rangex1/2.0],linewidth=2,color='k',linestyle='--')
plt.plot([rangex0/2.0,-rangex0/2.0], [-rangex1/2.0,-rangex1/2.0],linewidth=2,color='k',linestyle='--')
plt.plot([-rangex0/2.0,-rangex0/2.0], [-rangex1/2.0,rangex1/2.0],linewidth=2,color='k',linestyle='--')
#    
plt.hold(False)    
plt.grid('on')
plt.axis('square')
plt.xlim(mxu*array([-1.0,1.0]))
plt.ylim(mxu*array([-1.0,1.0]))
plt.title('dmin = %5.2f, dmax = %5.2f'%(dmin,dmax), fontsize=10)
plt.xticks(fontsize=8)
plt.xlabel('km',fontsize=10)
plt.yticks(fontsize=8)
plt.ylabel('km',fontsize=10)

plt.subplot(223)
plt.plot(sort(distance),'.-')
plt.grid('on')

plt.subplot(224)
plt.plot(sort(abs(orient))*180.0/pi,'.-')
plt.grid('on')
