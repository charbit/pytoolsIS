# -*- coding: utf-8 -*-
"""
Created on Thu May 26 06:44:15 2016

@author: maurice
"""

print '****************************'
print('\tNumber of runs to reach the conditions : %i'%(cr0))
print '\tR2_d = %3.2f, R2_o = %3.2f, RR = %3.2f, area = %5.3f\n\tdmin = %3.2f, dmax = %3.2f, Hd ratio = %3.2f, Ho ratio = %3.2f' \
     %(R2_d, R2_o, RR, areaCov, dmin, dmax, Hd_rel, Ho_rel)
print '****************************'

HorizontalSize = 8
VerticalSize   = 8

figOptimGeometry=plt.figure(num='CRB',figsize=(HorizontalSize,VerticalSize), 
               edgecolor='k', facecolor = [1,1,0.92]);


#%%
text2sfail = [];
cp=0
for im1 in range(M-1):
    for im2 in range(im1+1,M):
        aux = '%i:\t(%i, %i)' %(cp+1,pairf2[cp,0]+1,pairf2[cp,1]+1)
        cp=cp+1
        text2sfail.append(aux)
for cp in range(combi):
    print '%i\t: %s' %(cp+1,text2sfail[cp])
    
#====================
plt.figure(num='CRB')
Ckm = 1000.0
plt.clf()
plt.subplot(2,3,1)
for im in range(M):
    plt.plot(Ckm*x[im,0],Ckm*x[im,1],'o',markersize=9)
    plt.hold('on')
    plt.text(0.85*Ckm*x[im,0],0.85*Ckm*x[im,1],'%i'%(im+1))
    
plt.plot(Ckm*array([-rangex0/2.0,rangex0/2.0]), Ckm*array([rangex1/2.0,rangex1/2.0]),linewidth=2,color='k',linestyle='--')
plt.plot(Ckm*array([rangex0/2.0,rangex0/2.0]), Ckm*array([rangex1/2.0,-rangex1/2.0]),linewidth=2,color='k',linestyle='--')
plt.plot(Ckm*array([rangex0/2.0,-rangex0/2.0]), Ckm*array([-rangex1/2.0,-rangex1/2.0]),linewidth=2,color='k',linestyle='--')
plt.plot(Ckm*array([-rangex0/2.0,-rangex0/2.0]), Ckm*array([-rangex1/2.0,rangex1/2.0]),linewidth=2,color='k',linestyle='--')
#    
plt.hold('off')    
plt.grid('on')
plt.axis('square')

mxu= 1.1*Ckm*max([rangex0/2.0,rangex1/2.0])

plt.xlim(mxu*array([-1.0,1.0]))
plt.ylim(mxu*array([-1.0,1.0]))
plt.title('dmin = %5.2f, dmax = %5.2f'%(dmin,dmax), fontsize=10)
plt.xticks(fontsize=8)
plt.xlabel('km',fontsize=10)
plt.yticks(fontsize=8)
plt.ylabel('km',fontsize=10)

plt.subplot(4,3,2)
plt.plot(sort(distance))
plt.hold('on')
for im in range(M):
    plt.plot(sort(distance_f1[im,:]),'--')
    
plt.hold('off')
plt.grid('on')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('distances with R2-d = %5.2f'%R2_d, fontsize=10)
    
plt.subplot(4,3,5)
plt.plot(sort(abs(orient))*180.0/pi)
plt.hold('on')
for im in range(M):
    plt.plot(sort(abs(orient_f1[im,:]))*180.0/pi,'--')
    
plt.hold('off')

plt.ylim([0,180])
plt.grid('on')
plt.xticks(fontsize=8)
plt.yticks(arange(0,180.1,60),fontsize=8)
plt.ylabel('degree',fontsize=8, horizontalalignment='right')
plt.title('orientations with R2-o = %5.2f'%R2_o, fontsize=10)


plt.subplot(4,3,7)
plt.plot(arange(1,M+1),RR_f1,'.-')
plt.grid('on')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.hold ('on')
plt.plot(array([1,M]),RR*array([1.0,1.0]),'--k')
plt.hold ('off')
plt.xlim([1,M])
plt.title('isotropy with 1 sensor fail',fontsize=10)


plt.subplot(4,3,10)
meanSTDCRBaz_deg = mean(STDCRBaz_deg)
meanSTDCRBazf1_deg = mean(STDCRBazf1_deg,axis=1)
plt.plot(arange(1,M+1),(meanSTDCRBazf1_deg/meanSTDCRBaz_deg),'.-')
plt.grid('on')
plt.xlim([1,M])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('one failing sensor index')
plt.title('accuracy with 1 sensor fail',fontsize=10)

plt.subplot(4,3,8)
plt.plot(arange(1,combi+1),RR_f2,'.-')
plt.hold ('on')
plt.plot(array([1,combi]),RR*array([1.0,1.0]),'--k')
plt.xlim([1,combi])
plt.hold ('off')
plt.grid('on')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('isotropy with 2 sensor fails',fontsize=10)


meanSTDCRBazf2_deg = mean(STDCRBazf2_deg,axis=1)
plt.subplot(4,3,11)
plt.plot(arange(1,combi+1),(meanSTDCRBazf2_deg/meanSTDCRBaz_deg),'.-')
plt.xlim([1,combi])
plt.grid('on')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('accuracy with 2 sensor fails',fontsize=10)
plt.xlabel('two failing sensor index')
plt.subplot(1,3,3)
plt.ylim([-(combi+3), 1])
plt.xlim([0,1])
for cp in range(combi):
    plt.text(0.3,-(cp+2),text2sfail[cp])
plt.xticks([])
plt.yticks([])
plt.box('off')

plt.title('pair index')
plt.show()
