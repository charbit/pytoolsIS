# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:53:26 2016

@author: maurice
"""
from numpy import isnan, nan, zeros, real, imag, conjugate, transpose, dot, log
from scipy.linalg import sqrtm, inv

#%%
def trimmeancomplex(z,trimpercent,visu=0):
# synospsis 
#     trim = trimmeancomplex(z,trimpercent)
# inputs: 
#         z: is an 1d array of length N of COMPLEX values
#         trimpercent: a value in (0,1)
#                if trimpercent=1, we have the trivial mean
# output:
#         trim: returns the trimmed mean of the complex values 
#================================================== 
    from numpy import nanmedian, sqrt, size, nanmean
    from numpy import copy
    from matplotlib import pyplot as plt
    c = -2*log(1.0-trimpercent);
    N = len(z);
    Nnotnan = N-sum(isnan(z));
    trimz = zeros(size(z),dtype=complex)
    print Nnotnan
    cp = 0;
    if Nnotnan > 20:
        zt = copy(z);
        zt[isnan(zt)] = 0.0;
        meanz = sum(zt)/Nnotnan;
        zc = zt-meanz * (1-(zt==0));
        zri      = zeros([2,N])
        zri[0,:] = real(zc);
        zri[1,:] = imag(zc);    
        R = dot(zri,transpose(zri))/(Nnotnan-1);
        
        Fm1 = inv(R);
        
        for ii in range(N):
            zri_ii = zri[:,ii];
            if dot(zri_ii.reshape(1,2), 
                   dot(Fm1, zri_ii.reshape(2,1))) < c \
                   and not(isnan(z[ii]).real)\
                   and not(isnan(z[ii]).imag):
                trimz[ii] = z[ii];
                cp=cp+1;
            else:
                trimz[ii] = nan + 1j*nan
        print visu
        if visu:
            print R
            mytrim = nanmean(trimz)
            medz = nanmedian(z)
            y = ellipse(meanz,R,c)

            plt.plot(z.real,z.imag,'xb')
            plt.hold('True')
#            plt.plot(zri[0,:],zri[1,:],'d')
            #plt.plot(zc.real,zc.imag,'xk')
            
            plt.plot(mytrim.real,mytrim.imag,'or')
            plt.plot(medz.real,medz.imag,'om')
            plt.plot(meanz.real,meanz.imag,'oy')
    
            plt.plot(trimz.real,trimz.imag,'dk')
            plt.plot(y.real,y.imag)
            plt.hold('False')
            plt.title('%i,%i' %(N,Nnotnan))
            plt.show()
            
    else:
        trimz = copy(z);
        
    return trimz
    
def ellipse(m0,Gamma,c):
    from numpy import sqrt, pi, arange, exp
    E = sqrtm(Gamma)
    x = sqrt(c)*exp(2j*pi*arange(100)/99);
    y = dot(E, [x.real,x.imag])
    z = y[0,:]+1j*y[1,:]+m0
    return z
    