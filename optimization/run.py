
import logging
import logging.config
import datetime
import numpy
from numpy import *
import scipy
import scipy.sparse
import scipy.io



import matplotlib
#import socket
#if socket.gethostname() == "kinross":
#  matplotlib.use('pdf')

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt

from pegasos import pegasos
from pointsaga2 import pointsaga2
from pointsaga1 import pointsaga1
from saga import saga
from sdca import sdca
from csdca import csdca

import time


linestyles = ['-', '--', ':', '-.', '-', '--', '-.', ':',
              '-', '--', ':', '-.', '-', '--', '-.', ':',
              '-', '--', ':', '-.', '-', '--', '-.', ':']
colors = ['k', 'r', 'b', 'g', 'DarkOrange', 'm', 'y', 'c', 'Pink',
          '#00FF00', "#800000", "#808000", "#008080", "#B8860B",
          "#BDB76B", "#66CDAA", "#8A2BE2", "#8B4513"]

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("opt")
sTime = time.time()

random.seed(42)

logger.info("Loading data")
#dataset = scipy.io.loadmat("australian_scale.mat")
#dataset = scipy.io.loadmat("mushrooms.mat")
dataset = scipy.io.loadmat("rcv1_train.binary.mat")
X = dataset['X'].transpose()
d = dataset['d'].flatten()

n = X.shape[0]
m = X.shape[1]
logger.info("Data with %d points and %d features loaded.", n, m)

logger.info("Train Proportions: -1 %d   1: %d", sum(d == -1.0), sum(d == 1.0))

def runit():
    if True:
    
        #reg =  0.000001
        #reg =  0.000001
        #result = pointsaga2(X, d, {'useLag':True, 'stepSize': 0.5, 'passes': 20})
        #result = pointsaga2(X, d, {'useHinge': False, 'useLag':False, 'stepSize': 16.0, 'passes': 20, "reg": reg})
    
        #result = pointsaga2(X, d, {'useHinge': False, 'useLag':True, 'stepSize': 0.5, 'passes': 20, "reg": 0.00001})
    
#INFO:pointsaga1:Epoch 2 finished
#INFO:logisticloss:wbar: loss: 0.258832592358 errors: 625 (3.088 percent)
#INFO:logisticloss:x: loss: 0.258876445360 errors: 602 (2.974 percent)
        
       # result = pointsaga1(X, d, {'loss': 'logistic', 'stepSize': 2.0, 'averageAfter': 1,
        #    'passes': 3, "reg": 0.0001, 'returnAveraged': True, 'useLag': True})
        
        #result = pointsaga2(X, d, {'loss': 'logistic', 'stepSize': 32.0, 
        #    'passes': 20, "reg": 0.0001})
        #INFO:squaredloss: loss: 0.192206
        #result = saga(X, d, {'loss': 'squared', 'stepSize': 0.5, 'passes': 50, "reg": 0.001})
        
    
        # reg =  0.0001. Minimum at 0.258275
        #result = saga(X, d, {'useHinge': False, 'stepSize': 0.5, 'passes': 20, "reg": 0.0001})
    
    
        #result = pointsaga2(X, d, {'useLag':False})

        #alpha_k: 1.62639e-03, beta_k: 1.10803e+00, mu: 4.94020e-06
        #result_pegasos = pegasos(X, d, {'passes': 10, 'useHinge': False, 
        #    'stepSize': 64.0, 'reg': 0.0001, 'returnAveraged': True})

        # Compare lots of different step sizes.


        #Loss: 0.258275346488 errors: 569 (2.811 percent)
        #result = pointsaga1(X, d, {'loss': 'logistic', 'passes': 40, "reg": 0.0001})
        result = pointsaga2(X, d, {'loss': 'logistic', 'passes': 40, "reg": 0.0001})
        #result = csdca(X, d, {'loss': 'logistic', 'passes': 50, "reg": 0.0001, 'stepSize': 0.001})
        
        #$result = sdca(X, d, {'loss': 'logistic', 'passes': 20, "reg": 0.001})

    if False:
        result = result_lag
        #result = result_pegasos
        # Plot

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        #ax1.set_yscale('log')
        ax1.set_ylabel("Function suboptimality")
        ax1.set_xlabel("Epoch")
        ax1.set_ylim([0,0.5])

        pp = range(1, 1+len(result["flist"]))
        ax1.plot(pp, result["flist"], linestyle=linestyles[1], color=colors[1])

        #ax1.legend(captions, 'upper right')

        fname = "plot1"
        logger.info("Saving plot: %s", fname)
        fig1.savefig("plots/%s.pdf" % fname)

    if False:
        against_fmin = False
    
        passes = 20
        stepSizes = [100, 16, 8, 4, 2, 0.9, 0.5]
        #stepSizes = [0.9, 0.5, 0.1, 0.05, 0.01, 0.001]

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        ax1.set_ylabel("Function suboptimality")
        ax1.set_xlabel("Epoch")
    
        #ax1.set_ylim([0,0.5])

    
        if against_fmin:
            ax1.set_yscale('log')
            props = {'useLag':True, 'stepSize': 0.5, 'passes': 200}
            result = pointsaga2(X, d, props)
            fmin = result["flist"][-1]

        pp = range(1, 1+passes)
        captions = []
    
        idx = 0
        for stepSize in stepSizes:
            props = {'useLag':True, 'stepSize': stepSize, 'passes': passes}
            # Plot for a sweep of regularisation constants
            #result = pointsaga2(X, d, props)
            result = pegasos(X, d, props)
            captions.append("gamma=%1.2e" % stepSize)

            flist = result["flist"]
        
            if against_fmin:
                flist = [fi - fmin for fi in flist]

            ax1.plot(pp, flist, linestyle=linestyles[idx], color=colors[idx])
            idx = idx + 1
    
        ax1.legend(captions, loc='upper right')

        fname = "stepsizesweep1"
        logger.info("Saving plot: %s", fname)
        fig1.savefig("plots/%s.pdf" % fname)
    
    
    # Compare methods
    if False:
        passes = 40

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        ax1.set_ylabel("Function suboptimality")
        ax1.set_xlabel("Epoch")
    
        #ax1.set_ylim([0,0.5])

        ax1.set_yscale('log')
        props = {'useLag':True, 'stepSize': 0.5, 'passes': 200}
        result = pointsaga2(X, d, props)
        fmin = result["flist"][-1]

        pp = range(1, 1+passes)
        captions = []
    
        idx = 0
    
        methods = ["Pegasos", "Point-SAGA"]
        stepSize = {"Pegasos": 0.9, "Point-SAGA": 0.5}
        runFunc = {"Pegasos": pegasos, "Point-SAGA": pointsaga2}
    
        for method in methods:
            props = {'useLag':True, 'stepSize': stepSize[method], 'passes': passes}
            # Plot for a sweep of regularisation constants
            #result = pointsaga2(X, d, props)
            result = runFunc[method](X, d, props)
            captions.append(method)

            flist = result["flist"]
            flist = [fi - fmin for fi in flist]

            ax1.plot(pp, flist, linestyle=linestyles[idx], color=colors[idx])
            idx = idx + 1
    
        ax1.legend(captions, loc='lower left')

        fname = "Comparison1"
        logger.info("Saving plot: %s", fname)
        fig1.savefig("plots/%s.pdf" % fname)

if __name__ == "__main__":
    runit()
