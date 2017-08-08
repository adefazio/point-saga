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
def cgsvrg(A, double[:] b, props):

    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, p, epoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, cold, activation, cchange, gscaling, sg, new_loc, uk_sq, uknew_sq
    cdef double gr_dot_xk, grHgr, grHuk, alpha, gr_sq, gr_dot_gnew, conj

    # Data points are stored in columns in CSC format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr

    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions

    # Current iterate
    cdef double[:] xk = np.zeros(m)
    cdef double[:] uk = np.zeros(m)
    # Prox center
    cdef double[:] yk = np.zeros(m)
    # SVRG recalibratino point
    cdef double[:] zk = np.zeros(m)
    # Full gradient at zk (without reg)
    cdef double[:] gk = np.zeros(m)
    # Full gradient with reg
    cdef double[:] gr = np.zeros(m)
    # Hessian vector product against gkr
    cdef double[:] Hgr = np.zeros(m)

    cdef double gamma = props.get("stepSize", 0.1)
    cdef double reg = props.get('reg', 0.0001)

    cdef ninv = 1.0/n

    loss = getLoss(A, b, props)

    cdef double wscale = 1.0 # Scaling factor for xk.
    cdef double ry, mu

    np.random.seed(42)

    logger = logging.getLogger("cgsvrg")

    cdef int maxiter = props.get("passes", 10)

    cdef unsigned int k = 0 # Current iteration number

    cdef long [:] perm = np.random.permutation(n)

    xlist = []
    flist = []
    errorlist = []

    logger.info("reg: %1.3e, gamma: %1.3e", reg, gamma)
    logger.info("cgsvrg starting, npoints=%d, ndims=%d" % (n, m))

    loss.store_training_loss(xk)

    for epoch in range(maxiter):
      #  Recalculate the recalibration point's gradient at zk=zk
      for p in range(m):
        zk[p] = xk[p]
        gk[p] = 0.0
        Hgr[p] = 0.0

      for i in range(n):
        indstart = indptr[i]
        indend = indptr[i+1]
        ydata = data[indstart:indend]
        yindices = indices[indstart:indend]
        ylen = indend-indstart
        activation = spdot(xk, ydata, yindices, ylen)
        cnew = loss.subgradient(i, activation)
        add_weighted(gk, ydata, yindices, ylen, cnew*ninv)

      for p in range(m):
          gr[p] = gk[p] + reg*xk[p]

      # Compute hessian-vector product against gradient
      # X^T * X * gr
      for i in range(n):
          gr_dot_xk = spdot(gr, ydata, yindices, ylen)
          add_weighted(Hgr, ydata, yindices, ylen, ninv*gr_dot_xk)

      grHgr = 0.0
      gr_sq = 0.0
      for p in range(m):
          # regularisation bit
          Hgr[p] += reg*gr[p]
          grHgr += Hgr[p] * gr[p]
          gr_sq += gr[p] * gr[p]

      alpha = -gr_sq/grHgr

      # Take a global step before starting the stochastic steps.
      for p in range(m):
          xk[p] += alpha*gr[p]

      if True:
          # Check that gradient at new xk is conjugate to gr
          gr_dot_gnew = 0.0
          for p in range(m):
            # regularisation bit
            gr_dot_gnew += gr[p] * reg*xk[p]

          for i in range(n):
              indstart = indptr[i]
              indend = indptr[i+1]
              ydata = data[indstart:indend]
              yindices = indices[indstart:indend]
              ylen = indend-indstart
              ry = b[i]
              activation = spdot(xk, ydata, yindices, ylen)
              cnew = loss.subgradient(i, activation)
              gr_dot_gnew += ninv*cnew*spdot(gr, ydata, yindices, ylen)
              #add_weighted(gk, ydata, yindices, ylen, cnew*ninv)
          logger.info("gr dot gnew: %2.2f, alpha: %2.3f" % (gr_dot_gnew, alpha))

      ##################
      for j in range(n):

          if epoch == 0:
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

          activation = spdot(xk, ydata, yindices, ylen)
          cnew = loss.subgradient(i, activation)

          activation = spdot(zk, ydata, yindices, ylen)
          cold = loss.subgradient(i, activation)

          #SVRG step:
          # Part of the step is dense:

          for p in range(m):
            uk[p] = reg*xk[p] + gk[p]
            #xk[p] = (1-gamma*reg)*xk[p] - gamma*gk[p]
          # Sparse part:
          add_weighted(uk, ydata, yindices, ylen, (cnew-cold))

          # Conjugate it
          grHuk = 0.0
          uk_sq = 0.0
          for p in range(m):
              grHuk += Hgr[p] * uk[p]
              uk_sq += uk[p] * uk[p] # compute norm_sq as well

          alpha = grHuk/grHgr
          for p in range(m):
              uk[p] = uk[p] - alpha*gr[p]
              uknew_sq += uk[p] * uk[p]

          #alpha = sqrt(uk_sq/uknew_sq) # want the steps norm the match the old one

          for p in range(m):
              xk[p] = xk[p] - gamma*uk[p]

          if False:
              # Check conjugacy
              conj = 0.0
              uk_sq = 0.0
              for p in range(m):
                  conj += Hgr[p] * uk[p]
                  uk_sq += uk[p] * uk[p]
              logger.info("conj: %2.2f uk_sq: %2.2f" % (conj, uk_sq))


      loss.store_training_loss(xk, "outer xk:")

    logger.info("cgsvrg finished")

    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
