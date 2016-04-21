
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
from wsaga import wsaga

import time

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("opt")
sTime = time.time()

random.seed(42)

logger.info("Loading data")
dataset = scipy.io.loadmat("australian.mat")
#dataset = scipy.io.loadmat("australian_scale.mat")
#dataset = scipy.io.loadmat("mushrooms.mat")
#dataset = scipy.io.loadmat("rcv1_train.binary.mat")
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
    
    #result = saga(X, d, {'loss': 'logistic', 'passes': 30, "reg": 0.0001})
    #result = lsaga(X, d, {'loss': 'logistic', 'passes': 30, "reg": 0.0001})
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
    
    #result = wsaga(X, d, {'loss': 'logistic', 'passes': 50, "reg": 0.0001})
    # Ok, version where we decouple table update from step seems to work. good.
    # Now to add the proportional sampling.
    
    # The weighted sampling variant seems to be having issues though.
if __name__ == "__main__":
    runit()
