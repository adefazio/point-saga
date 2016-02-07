#cython: profile=False
import random
import numpy as np
cimport numpy as np

cimport cython
from cython.view cimport array as cvarray

import logging
import scipy
from scipy.sparse import csr_matrix
from libc.math cimport exp, sqrt, log, fabs

from cpython cimport bool

from hinge_loss import HingeLoss
from logistic_loss import LogisticLoss
from squared_loss import SquaredLoss

STUFF = "Hi"
def getLoss(Xn, dn, props):
  loss_name = props.get("loss", "logistic")
  if loss_name == "logistic":
    return(LogisticLoss(Xn, dn, props))
  if loss_name == "hinge":
    return(HingeLoss(Xn, dn, props))
  if loss_name == "squared":
    return(SquaredLoss(Xn, dn, props))

  raise Exception("Loss not recognised: %s" % loss_name)
