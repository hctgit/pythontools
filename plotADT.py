#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path
from matplotlib.colors import LogNorm
from math import factorial
from scipy.stats.stats import pearsonr   
import os,sys

def xy(r,phi):
  return r*np.cos(phi), r*np.sin(phi)

def emit(x,p):
	return	np.std(x)*np.std(p)*sqrt(1 - pearsonr(x,p)[0])

def dataAcquisition(fileData):
    fileDType1 = np.dtype([('partID', np.int), ('turn', np.int),
                       ('x', np.float), ('xp', np.float),
                       ('amplitude', np.float), ('tune_ADT', np.float)])
    data = np.loadtxt(fileData,fileDType1)
    return data

def getPhaseSpace(filename):
    fdata = dataAcquisition(filename)
    x = fdata[:]["x"]
    xp = fdata[:]["xp"]
    return x, xp

def plotAmpHist(x,xp,fig):
    #rc('text', usetex=True)
    #rc('xtick', labelsize=15) 
    #rc('ytick', labelsize=15)
    plt.figure(fig)
    hist = plt.hist(np.sqrt(x**2+xp**2), bins=50)
    #plt.xlabel(r"$(X^2+P^2)^{1/2}$",fontsize=22)
    #plt.ylabel(r"Counts",fontsize=22)
    #plotAmpHist = plt.show()
    return hist

def plotRamp(fname):
    fileDType2 = np.dtype([('turn', np.int),('tune_ADT', np.float), ('amplitude_ADT', np.float)])
    fdata_ramp = np.loadtxt(fname, dtype=fileDType2)
    plt.plot(fdata_ramp[:]['turn'], fdata_ramp[:]['tune_ADT'])
    plotRamp = plt.show()
    return plotRamp

def plotHist2D(x,xp):
    plotHist2D = plt.hist2d(x,xp,bins=200,norm=LogNorm())
    plt.colorbar()
    #plt.ylabel(r"$P_x$",fontsize=15)
    #plt.xlabel(r"$X$",fontsize=15)
    #plotHist2D = plt.show()
    return plotHist2D

def checkParticleLoss(x,xp,cut=3,count=0):
    Npart = len(x)
    xInCut = []
    xpInCut = []
    for i in range(Npart):
        if (np.sqrt(x[i]**2+xp[i]**2) > cut):
            count += 1
        else:
            xInCut.append(x[i])
            xpInCut.append(xp[i])
    return count, xInCut, xpInCut

def emittance(x,xp):
    emit = np.std(x)*np.std(xp)*np.sqrt(1 - pearsonr(x,xp)[0])
    return emit
