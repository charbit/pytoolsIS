# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 07:52:32 2016

@author: maurice
"""

def trimmeancomplex(z,trimpercent):
# synospsis 
#     trim = trimmeancomplex(z,trimpercent)
# inputs: 
#         z: is an 1d array of length N of COMPLEX values
#         trimpercent: a value in (0,1)
#                if trimpercent=1, we have the trivial mean
# output:
#         trim: returns the trimmed mean of the complex values 
#================================================== 

    c = -2*log(1.0-trimpercent);
    N = len(z);
    Nnotnan = N-sum(isnan(z));
    trimz = zeros(size(z),dtype=complex)
   
    cp = 0;
    if Nnotnan > 20:
        zt = z.copy();
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
    else:
        trimz = z.copy();        
    return trimz
    