# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:02:41 2016

@author: maurice
"""




# function c = invcumulFunctionMSC(p,C0,Nd,precision)
def invcumulFunctionMSC(p,C0,Nd,precision=1e-5):
#=============================================================
# Inverse cumulative function of the MSC estimates
# derived from the spectral estimation
#            |S_xy|^2
#    MSC = ------------
#           S_xx S_yy
#=============================================================
# Synopsis
#          c = invcumulFunctionMSC(p,C0,Nd,precision)
# where
#   p
#   C0 : true coherence value in (0,1)
#   Nd : size of the smoothing window or averaging window
#   precision: default value 1e-15
#=============================================================
# outputs
#   P = Prob(\hat MSC =< E; N, MSC)
#   E 
#   CI : confidence interval at 100a%
#
# Rk: the resolution is obtained by dichotomy, whiic is easy
# because the cumulative function is by def monotonic.
#=============================================================    
    lastpos = 1.0;
    lastneg = 0.0;
    c       = 1.0;
    while abs(lastpos-lastneg)>precision:
        dy = cumulFunctionMSC(c,C0,Nd,p);
        if dy < 0:
            lastneg = c;
            c = (lastneg+lastpos)/2.0;
        else:
            lastpos = c;
            c = (lastneg+lastpos)/2.0;
        
    return c
#=============================================================
# function dP = cumulFunctionMSC(E,C0,Nd,p)
def cumulFunctionMSC(E,C0,Nd,p):
    z        = E*C0;
    sum_F    = 1.0;
    coef_ell = 1.0;
    
    R = ((1-C0) / (1-z)) ** Nd;
    for ell in range(1,Nd-1):
        Tk      = 1.0;
        sum_ell = 1.0;
        for k in range(1,ell+1):
            Tk = (k-1-ell)*(k-Nd)*(Tk*z)/k/k;
            sum_ell = sum_ell + Tk;
        
        coef_ell = coef_ell * ((1-E) / (1-z));
        sum_F = sum_F+coef_ell * sum_ell;
    
    dP = (R * sum_F) * E - p;
    return dP
#=============================================================

