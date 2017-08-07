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

#from hinge_loss import HingeLoss
#from logistic_loss import LogisticLoss

#from loss import getLoss

STUFF = "Hi"

ctypedef np.float32_t dtype_t

#cimport sparse_util
from sparse_util cimport spdot, add_weighted, lagged_update, lagged_update_with_xbar
from get_loss import getLoss

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def bregman(A, double[:] b, props):

    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, p, epoch, outerepoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, cold, activation, cchange, gscaling, sg, new_loc

    # Data points are stored in columns in CSC format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr

    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions

    cdef double[:] xk = np.zeros(m)
    cdef double[:] xk_old = np.zeros(m)
    cdef double[:] yk = np.zeros(m)
    cdef double[:] yk_old = np.zeros(m)
    cdef double[:] yk_prime = np.zeros(m)
    cdef double[:] zk = np.zeros(m)
    cdef double[:] gk = np.zeros(m)

    cdef double gamma = props.get("stepSize", 0.1)
    cdef double reg = props.get('reg', 0.0001)
    #cdef bool use_lag = props.get("useLag", True)
    cdef bool use_perm = props.get("usePerm", False)

    cdef ninv = 1.0/n

    loss = getLoss(A, b, props)

    cdef double wscale = 1.0 # Scaling factor for xk.
    cdef double ry, mu

    np.random.seed(42)

    logger = logging.getLogger("bregman")

    cdef int maxiter = props.get("passes", 10)
    cdef int maxinner = props.get("maxinner", 1)

    cdef unsigned int k = 0 # Current iteration number

    cdef long [:] perm = np.random.permutation(n)

    xlist = []
    flist = []
    errorlist = []

    logger.info("reg: %1.3e, gamma: %1.3e", reg, gamma)
    logger.info("Bregman starting, npoints=%d, ndims=%d" % (n, m))

    loss.store_training_loss(xk)

    for outerepoch in range(maxiter):
      #  Recalculate the recalibration point's gradient at zk=zk
      for p in range(m):
        zk[p] = xk[p]
        gk[p] = 0.0

      for i in range(n):
        indstart = indptr[i]
        indend = indptr[i+1]
        ydata = data[indstart:indend]
        yindices = indices[indstart:indend]
        ylen = indend-indstart
        ry = b[i]
        activation = spdot(xk, ydata, yindices, ylen)
        cnew = loss.subgradient(i, activation)
        add_weighted(gk, ydata, yindices, ylen, cnew*ninv)

      for epoch in range(maxinner):

          for j in range(n):

              if epoch == 0 and outerepoch == 0:
                  i = perm[j]
              else:
                  i = np.random.randint(0, n)

              indstart = indptr[i]
              indend = indptr[i+1]
              ydata = data[indstart:indend]
              yindices = indices[indstart:indend]
              ylen = indend-indstart
              ry = b[i]

              k += 1

              activation_new = spdot(xk, ydata, yindices, ylen)
              activation_old = spdot(zk, ydata, yindices, ylen)

              cnew = loss.subgradient(i, activation_new)
              cold = loss.subgradient(i, activation_old)


              #SVRG step:
              # Part of the step is dense:
              for p in range(m):
                xk[p] = (1-gamma*reg)*xk[p] - gamma*gk[p]
              # Sparse part:
              add_weighted(xk, ydata, yindices, ylen, -(cnew-cold)*gamma)

          logger.info("Bregman Inner epoch finished")

      # Set yk
      for p in range(m):
        yk[p] = xk[p]
        xk_old[p] = xk[p]

      #alpha = alpha_new

      # Set new xk
      #for p in range(m):
      #    yk[p] = reg_conversion_factor*yk[p]
      #    xk[p] = xk[p] - yk_old[p] + yk[p]
      #    yk_old[p] = yk[p]

      #loss.compute_loss(xk, "xk:")
      loss.store_training_loss(xk_old, "xk:")

    logger.info("Bregman finished")

    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
