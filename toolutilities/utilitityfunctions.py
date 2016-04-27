# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 07:00:13 2016

@author: maurice
"""

import inspect
import sys
sys.path.append('ISprocess')
sys.path.append('toolbenchmark')


def functionlistmodule(modulename):
    """
    synposis: functionlistmodule(modulename)
    where modulename is a name of a module
    Rk: before use we must import the module
    do
    
       - import sys
       - sys.path.append('ISprocess') <-- (directory of'modulename')
       - from listmyfunctions import functioninprocessIS
       - import the module called 'modulename'
       - functioninprocessIS(modulename)
    """
    listfunctions=inspect.getmembers(modulename, inspect.isfunction)
    laa=len(listfunctions)
    L=[]
    for i in range(laa):
        pL = inspect.getmodule(listfunctions[i][1])
        if pL == modulename:
            #print listfunctions[i][0]
            L = L+[listfunctions[i][0]]
    return L
    
