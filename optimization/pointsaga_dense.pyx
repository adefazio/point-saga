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

STUFF = "Hi"

ctypedef np.float32_t dtype_t

#cimport sparse_util
from sparse_util cimport spdot, add_weighted, lagged_update
from get_loss import getLoss

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def pointsaga_dense(A, double[:] b, props):

    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, p, epoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, cold, sg, new_loc

    # Data points are stored in columns in CSC format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr

    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions
    cdef unsigned int nseen = 0

    cdef double[:] xk = np.zeros(m)
    cdef double[:] gk = np.zeros(m)

    cdef double gamma = props.get("stepSize", 0.1)
    cdef double reg = props.get('reg', 0.0001)
    cdef bool use_lag = props.get("useLag", True)
    cdef bool use_perm = props.get("usePerm", False)

    loss = getLoss(A, b, props)

    cdef double ry

    np.random.seed(42)

    logger = logging.getLogger("pointsaga dense")

    cdef int maxiter = props.get("passes", 10)

    # Used to convert from prox_f to prox of f plus a regulariser.
    cdef double gamma_prime
    cdef double prox_conversion_factor = 1-(reg*gamma)/(1+reg*gamma)

    # gradient table
    cdef double[:] c = np.zeros(n)

    cdef unsigned int k = 0 # Current iteration number

    cdef long [:] perm = np.random.permutation(n)

    xlist = []
    flist = []
    errorlist = []

    logger.info("Gamma: %1.2e, reg: %1.3e, prox_conversion_factor: %1.8f, 1-reg*gamma: %1.8f",
                gamma, reg, prox_conversion_factor, 1.0 - reg*gamma)
    logger.info("Point-saga starting, npoints=%d, ndims=%d" % (n, m))

    loss.store_training_loss(xk)

    for epoch in range(maxiter):

        for j in range(n):
            if epoch == 0:
                i = j
            else:
              if use_perm:
                if epoch % 2 == 0:
                  i = j
                else:
                  i = perm[j]
              else:
                i = np.random.randint(0, n)

            # Selects the (sparse) column of the data matrix containing datapoint i.
            indstart = indptr[i]
            indend = indptr[i+1]
            ydata = data[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart
            ry = b[i]

            k += 1

            gamma_prime = gamma*prox_conversion_factor

            #Apply old gradient from table
            add_weighted(xk, ydata, yindices, ylen, c[i]*gamma)

            for p in range(m):
                xk[p] = xk[p] - gamma*gk[p]
                xk[p] = prox_conversion_factor*xk[p]

            activation = spdot(xk, ydata, yindices, ylen)
            (new_loc, cnew) = loss.prox(gamma_prime, i, activation)

            cold = c[i]
            cchange = cnew-c[i]
            c[i] = cnew

            # Update xk with the f_j^prime(phi_j)
            add_weighted(xk, ydata, yindices, ylen, -cnew*gamma_prime)

            # update the gradient average
            add_weighted(gk, ydata, yindices, ylen, cchange/n)

            if epoch == 0:
              nseen = nseen + 1

        logger.info("Epoch %d finished", epoch)


        loss.store_training_loss(xk)

    logger.info("Point-saga finished")

    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
