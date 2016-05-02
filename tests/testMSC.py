# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:25:57 2016

@author: maurice
"""

# warning: for CI we have to take a/2 and 1-a/2
# for monolateral test we have to take 1-a
#

import sys
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolutilities')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/myjob/progspy/toolcalibration')

from geoloc import extractstationlocations

from invcumulMSC import invcumulFunctionMSC
from toolIS import maxfstat, synthetizer
from toolIS import evalCRBwithgaussianLOC, CRBonazimuthonlywithoutLOC

from numpy import linspace, zeros, array, ones, arange

from matplotlib import pyplot as plt

listC0      = arange(0,1,0.01);
listNd      = array([5,6,7,8]);
LC          = len(listC0);
LN          = len(listNd);
a           = 0.05;
CI          = zeros([LC,3,LN]);
for iN in range(LN):
    Nd = listNd[iN];
    for ic in range(LC):
        C0 =listC0[ic];
        CI[ic,0,iN]=invcumulFunctionMSC(a/2,C0,Nd);
        CI[ic,1,iN]=invcumulFunctionMSC(1-a/2,C0,Nd);
        CI[ic,2,iN]=invcumulFunctionMSC(1-a,C0,Nd);
    

#%%
#figure(1)


HorizontalSize = 4
VerticalSize   = 4
figMSC=plt.figure(num=2,figsize=(HorizontalSize,VerticalSize), edgecolor='k', facecolor = [1,1,0.92]);
figMSC.clf()

hs = plt.plot(listC0,CI[:,2,:])
plt.grid(True)
plt.hold(True)
MSC0indselect=90
for im in range(LN):
    tt = '$N_d$ = %i' %listNd[im]
    h=plt.plot(listC0[MSC0indselect],CI[MSC0indselect,2,im],'o', 
               label=tt);
    cc = plt.get(hs[im],'color');
    plt.setp(h,color=cc)
    if im==0:
        ha=plt.arrow(listC0[MSC0indselect]+0.01,CI[MSC0indselect,2,im]+0.01,
                  -0.005,-0.005, head_width=0.004)

plt.hold(False)
plt.xlim([0.85, 0.93])
plt.ylim([0.89, 1])
plt.legend(loc='best')

dirfigsave = '/Users/maurice/etudes/ctbto/allJOBs2015/reponsesdevelop/rep2'
tt='%s/MSCthreshold2.pdf' %dirfigsave
figMSC.savefig(tt,format='pdf')


#grid on
#ylabel('\eta')
#xlabel('C')
#legend(sprintf('N_d = %i',listNd(1)), ...
#    sprintf('N_d = %i',listNd(2)), ...
#    sprintf('N_d = %i',listNd(3)), ...
#    sprintf('N_d = %i',listNd(4)))
#
#HorizontalSize = 18;
#VerticalSize   = 12;
#set(gcf,'units','centimeters');
#set(gcf,'paperunits','centimeters');
#set(gcf,'PaperType','a4');
## set(gcf,'position',[0 5 HorizontalSize VerticalSize]);
#set(gcf,'paperposition',[0 0 HorizontalSize VerticalSize]);
#set(gca,'fontname','times','fontsize',10)
#
#set(gcf,'color', [1,1,0.92]);%0.7*ones(3,1))
#set(gcf, 'InvertHardCopy', 'off');
#
##  print -depsc -loose  ../texte2/thresholdforC95.eps
#
##===
#
#listthreshold = (0:0.05:1)';
#LTh           = length(listthreshold);
#ROC           = zeros(LTh, LN,LC,2);
#
#for iN = 1:LN
#    Nd = listNd(iN);
#    for ic = 1: LC
#        C0=listC0(ic);
#        ROC(:,iN,ic,1) = cumulFunctionMSC(listthreshold,0,Nd);
#        ROC(:,iN,ic,2) = cumulFunctionMSC(listthreshold,C0,Nd);
#    end
#end
