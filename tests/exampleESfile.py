# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:51:56 2016

@author: maurice
"""
from numpy import savez, load
from numpy import random, amax

x = random.randn(12,3)
z = random.randn(2,3)
savez('workfile.npz',a=x,b=z)
y=load('workfile.npz')
#==============
# 3 methods for the max of an array
print amax(abs(x-y['a']))

u=abs(x-y['a'])
print u.max()

print max(max(u, key=tuple))