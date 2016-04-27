# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:02:54 2016

@author: maurice
"""


class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
 
filtercharact = []

fil_item = struct(designname = 'butter',
       Norder         = 2,
       Wlow_Hz        = 0.01,
       Whigh_Hz       = 9.0,
       SCPperiod_sec  = 200,
       windowshape    = 'hann',
       overlapDFT     = 0.5,
       overlapSCP     = 0,
       ratioDFT2SCP   = 5)
filtercharact.append(fil_item)




