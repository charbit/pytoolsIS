#====================== estimationwithFBlite.m ===========================
# Program estimates SUT response from the signals
# located in the directory directorysignals.
# The signals correspond to the pair of sensors
# SUT/SREF during a given duration T, typically 48 hours.
# Here we concatene NBCONCAT randomly chosen files.
#
# The evaluated parameters consist of the ratios, the STDs
# They are obtained by averaging on the period T.
# Results are plotted in figure 1
#=============
# Here we use the structure FILTERCHARACT and the list of frequencies.
# The only useful processing is the call to the function 
#              ESTIMSUTLITE.M
#=============  IMPORTANT: list of inputs to the developper ==============
# - filter.num and .den
# - allfreqsinfilter_Hz
# - Fs_Hz
# - the signals (signals_centered variable in the following)
# - ref sensor response
#=========================================================================

import glob

import sys
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolbenchmark')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolboxIS')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolutilities')
sys.path.insert(0, '/Users/maurice/etudes/ctbto/allJOBs2016/progspy/toolcalibration')

from numpy import zeros, array
from numpy import log10
from numpy import concatenate, logspace
from numpy import random, ones

from numpy import disp, sort
from matplotlib import pyplot as plt
import scipy.io as io
from estimSUTlite import estimSUT 

class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)

# the following lines can be changed by the user:
MSCthreshold     = 0.9;
FLAGsavesmall    = 0;
Fs_Hz            = 20;
ihc              = 1;
trimpercent      = 0.7
nbrandomconcat   = 20;
frequencylist_Hz = logspace(-2,log10(6),30);
#===
# directories
directorysignals    = '../../../AAdataI26calib/';
#=========================================================================
#=========================================================================
#====== load the filter bank characteristics
#  the useful structure is FILTERCHARACT
filtercharactfilename = '../toolcalibration/filtercharacteristics1.py';
execfile(filtercharactfilename)
Pfilter = len(filtercharact)

dirdotmat = '/Users/maurice/etudes/ctbto/allJOBs2015/AAdataI26calib/'
ihc=1;
#===================== read data =========================
fileswithdotmat        = glob.glob(dirdotmat+'s%i'%ihc+'/s%iy*.mat'%ihc);
nbmats                 = len(fileswithdotmat);
signals                = list(zeros(nbrandomconcat));
alldates               = list(zeros(nbrandomconcat));

indperm                = random.randint(1,nbmats,nbmats);
selectedlist           = range(nbrandomconcat);#indperm[0:nbrandomconcat];
for indfile in range(nbrandomconcat):
    ifile              = selectedlist[indfile];
    fullfilename_i     = fileswithdotmat[ifile];
    yearlocation       = fullfilename_i.find('year');
    year               = fullfilename_i[yearlocation+4:yearlocation+8]
    monthlocation      = fullfilename_i.find('month');
    month              = fullfilename_i[monthlocation+5:monthlocation+7]
    daylocation        = fullfilename_i.find('day');
    day                = fullfilename_i[daylocation+3:daylocation+5]
    filenameonly       = fullfilename_i[yearlocation-2:yearlocation+24];
    alldates[indfile]  = [year+month+day]
    matfile            = io.loadmat(dirdotmat+'s%i'%ihc+'/%s'%filenameonly);
    signals[indfile]   = matfile['signals_centered']
    
ss = array(signals[0])
for indfile in range(1,nbrandomconcat):
    ss = concatenate((ss,array(signals[indfile])),axis=0)
    
disp('************************************************')
sortalldates = sort(alldates,0);
disp('Station %i:'%ihc);
disp(sortalldates)
##
disp('************* start process ********************')
#===============================================================
#===============================================================
#=============== processing function call ======================
#===============================================================

Rsup, freqslin, STDmoduleR, STDphaseR, nboverTH = \
       estimSUT(ss, filtercharact, frequencylist_Hz, \
       Fs_Hz, MSCthreshold, trimpercent);

##
#%%
#===============================================================
#================= for plotting ================================
#===============================================================
#===============================================================
plt.semilogx(freqslin, (abs(Rsup)),'or')
plt.hold('True')
#plt.semilogx(frequencylist_Hz, 1.5*ones(len(frequencylist_Hz)),'ob')
plt.hold('False')
plt.title('%i days' %(2*nbrandomconcat)+ ' on station %i' %ihc)
plt.xlabel('frequency - Hz')
plt.ylabel('Rsup - dB')
plt.grid()