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
def newton(A, double[:] b, props):

    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, p, epoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, cold, final_activation
    cdef double cx, rx, norm_sq, xactivation, step_weight, expact

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
    cdef bool use_perm = props.get("usePerm", False)

    loss = getLoss(A, b, props)

    cdef double wscale = 1.0
    cdef double ry

    # Tracks for each entry of x, what iteration it was last updated at.
    cdef unsigned int[:] lag = np.zeros(m, dtype='I')

    # Used to convert from prox_f to prox of f plus a regulariser.
    cdef double gamma_prime
    cdef double prox_conversion_factor = 1-(reg*gamma)/(1+reg*gamma)

    # This is just a table of the sum the geometric series (1-reg*gamma)
    # It is used to correctly do the just-in-time updating when
    # L2 regularisation is used.
    cdef double[:] lag_scaling = np.zeros(n+2)
    lag_scaling[0] = 0.0
    lag_scaling[1] = 1
    cdef double geosum = 1
    cdef double mult = prox_conversion_factor
    for i in range(2,n+2):
        geosum *= mult
        lag_scaling[i] = lag_scaling[i-1] + geosum

    np.random.seed(42)

    logger = logging.getLogger("newton sparse")

    cdef int maxiter = props.get("passes", 10)

    # gradient table
    cdef double[:] c = np.zeros(n)

    cdef unsigned int k = 0 # Current iteration number

    cdef long [:] perm = np.random.permutation(n)

    xlist = []
    flist = []
    errorlist = []

    logger.info("Gamma: %1.2e, reg: %1.3e, 1-reg*gamma: %1.8f",
                gamma, reg, 1.0 - reg*gamma)
    logger.info("Newton Point-saga starting, npoints=%d, ndims=%d" % (n, m))

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
            cold = c[i]

            k += 1

            xactivation = wscale*spdot(xk, ydata, yindices, ylen)
            cx = loss.subgradient(i, xactivation)
            expact = exp(ry*xactivation)
            rx = expact/((1.0+expact)**2)

            # Correct for lag:
            lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling, -gamma/wscale)

            #Apply gradient change to xk
            add_weighted(xk, ydata, yindices, ylen,
                         -gamma*(cx-cold-xactivation*rx)/wscale)

            #for p in range(m):
            #    xk[p] = xk[p] - gamma*gk[p]

            activation = wscale*spdot(xk, ydata, yindices, ylen)
            norm_sq = loss.normSq(i)

            # Scale xk down
            #for p in range(m):
            #    xk[p] = xk[p]/(1+gamma*reg)
            wscale *= prox_conversion_factor

            # This is the main equation really
            step_weight = gamma*rx*activation/((1+gamma*reg)**2)
            step_weight /= 1 + gamma*rx*norm_sq/(1+gamma*reg)

            add_weighted(xk, ydata, yindices, ylen, -step_weight/wscale)

            # Calculation for cnew
            final_activation = wscale*spdot(xk, ydata, yindices, ylen)
            cnew = cx + rx*(final_activation-xactivation)

            cchange = cnew-c[i]
            c[i] = cnew

            # update the gradient average
            add_weighted(gk, ydata, yindices, ylen, cchange/n)

            if epoch == 0:
              nseen = nseen + 1

        logger.info("Epoch %d finished", epoch)


        # Unlag the vector
        gscaling = -gamma/wscale
        for ind in range(m):
            lagged_amount = k-lag[ind]
            if lagged_amount > 0:
                lag[ind] = k
                xk[ind] += (lag_scaling[lagged_amount+1]-1)*gscaling*gk[ind]
            xk[ind] = wscale*xk[ind]

        wscale = 1.0

        loss.store_training_loss(xk)

    logger.info("Point-saga finished")

    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
