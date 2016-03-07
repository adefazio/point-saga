
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

import time

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
    
    #INFO:logisticloss: loss: 0.258285864622 errors: 574 (2.836 percent)
    #INFO:lsaga:Epoch 14 finished
    #INFO:logisticloss: loss: 0.258281887291
    
    #result = saga(X, d, {'loss': 'logistic', 'passes': 15, "reg": 0.0001})
    result = lsaga(X, d, {'loss': 'logistic', 'passes': 15, "reg": 0.0001, 'stepSize': 1.0/0.625})

if __name__ == "__main__":
    runit()
