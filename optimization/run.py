
import logging
import logging.config
import datetime
import numpy
from numpy import *
import scipy
import scipy.sparse
import scipy.io
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt

from pegasos import pegasos
from pointsaga import pointsaga
from saga import saga
from sdca import sdca
from csdca import csdca
from lsaga import lsaga
from isaga import isaga
from wsaga import wsaga
from sagafast import sagafast

import time

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("opt")
sTime = time.time()

random.seed(42)

logger.info("Loading data")
#dataset = scipy.io.loadmat("australian.mat")
#dataset = scipy.io.loadmat("australian_scale.mat")
#dataset = scipy.io.loadmat("mushrooms.mat")
#dataset = scipy.io.loadmat("covtype.libsvm.binary.scale.mat")
dataset = scipy.io.loadmat("rcv1_train.binary.mat") 
X = dataset['X'].transpose()
d = dataset['d'].flatten()

n = X.shape[0]
m = X.shape[1]
logger.info("Data with %d points and %d features loaded.", n, m)

logger.info("Train Proportions: -1 %d   1: %d", sum(d == -1.0), sum(d == 1.0))

def runit():
    
    #INFO:logisticloss: loss: 0.258285864622 errors: 574 (2.836 percent)
    #INFO:lsaga:Epoch 14 finished
    #INFO:logisticloss: loss: 0.258281887291
    
    result = sagafast(X, d, {'loss': 'logistic', 'passes': 50, "reg": 0.0001})
    
    
    #result = wsaga(X, d, {'loss': 'logistic', 'passes': 60, "reg": 0.0001, 'useSeparateUpdate': False})
    
    
    #result = isaga(X, d, {'loss': 'logistic', 'passes': 5, "reg": 0.0001})
    
    #result = isaga(X, d, {'loss': 'logistic', 'passes': 80, "reg": 0.0001, "stepSize": 4.0, "normalizeData": False, "regUpdatesPerPass": 50})
    
    #result = isaga(X, d, {'loss': 'logistic', 'passes': 50, "reg": 0.01, "stepSize": 4.0, "normalizeData": True, "regUpdatesPerPass": 100})
    
    #result = saga(X, d, {'loss': 'logistic', 'passes': 1000, "reg": 10.000, "stepSize": 2e-7})
    #result = lsaga(X, d, {'loss': 'logistic', 'passes': 30, "reg": 0.0001})
    #result = isaga(X, d, {'loss': 'logistic', 'passes': 1000, "reg": 1000.000, "adaptive": True, 'stepSize':16, 'regUpdatesPerPass': 200})
    #0.258275346457 
    #0.258275346457
    #0.258275346442
    #Australian non-scaled. Best step size 1.0/(5 000 000) , 'stepSize': 0.0000002
    
    # Unscaled point saga
    #result = pointsaga(X, d, {'loss': 'logistic', 'passes': 20, "reg": 0.0001})
    #result = pointsaga(X, d, {'loss': 'logistic', 'passes': 20, "reg": 0.0001, 'stepSize': 0.00001})
    # Point saga is not immune to the dimensional scaling problem.
    
    # At one point L is increased to 44,942,696. Yikes.
    #Lsaga, initial L: 370 508. Doesn't seem to be actually working.
    # Max norm squared in data: 10,000,403,362. 10 bilion. hm.
    
    # For australian_scaled norms are between 6.69 and 13.39.
    
    #result = saga(X, d, {'loss': 'logistic', 'passes': 50, "reg": 0.0001})
    #result = wsaga(X, d, {'loss': 'logistic', 'passes': 200, "reg": 0.0001, 'gammaScale': 0.5}) 
    #0.258275346400
    #0.258275346408
    # Ok, version where we decouple table update from step seems to work. good.
    # Now to add the proportional sampling.
    
    # The weighted sampling variant seems to be having issues though.
    
    # Normalizing transform for each column
   # Ld = pow(linalg.norm(X, axis=0), 2)/X.shape[0]
    # Apply it to X
    #Xn = multiply(X, Ld)
    
    
    #INFO:isaga:Percentiles of feature weights
    #INFO:isaga:[  4.60107006e-03   1.04801098e+00   2.07631943e+00   3.55215281e+00
    #   1.63313883e+01]
    #INFO:isaga:Squared norm percentiles [0, 25, 50, 75, 100] (After renorm):
    #INFO:isaga:[  2.28792303e-02   2.64173357e-01   5.86555276e-01   1.17757560e+00
    #   4.08003438e+01]
    #INFO:loss:Squared norm percentiles
    #INFO:loss:[ 2.  2.  2.  2.  2.  2.  2.]
    
if __name__ == "__main__":
    runit()
