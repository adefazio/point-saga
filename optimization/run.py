
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
from bregman import bregman
from cgsvrg import cgsvrg

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
    #reg 0.0001, expecting loss: 0.258275346XXX errors: 569 (2.811 percent)
    #"reg": 1e-6 step 1, 300 epochs: loss: 0.033747761546 errors: 22 (0.109 percent)

    #quadratic loss reg 5e-6. end loss 0.025039.
    #svrg end 0.025491, saga end  0.025473. Similar.
    result = cgsvrg(X, d, {'loss': 'squared', 'passes': 40, "reg": 5e-6, 'stepSize': 0.02})
    #result = pointsaga(X, d, {'loss': 'squared', 'passes': 40, "reg": 5e-6, 'stepSize': 1.0})

    #result = pointsaga(X, d, {'loss': 'logistic', 'passes': 40, "reg": 1e-6, 'stepSize': 1.0})
    #result = bregman(X, d, {'loss': 'logistic', 'passes': 20, "maxinner": 2, "reg": 1e-6, 'stepSize': 1.0,
    #                        "proxStrength": 0.0001, 'useBreg': True})

if __name__ == "__main__":
    runit()
