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
from libc.stdlib cimport malloc, free

from cpython cimport bool

from get_loss import getLoss

STUFF = "Hi"

ctypedef np.float32_t dtype_t

#cimport sparse_util
from sparse_util cimport spdot, add_weighted, lagged_update

ctypedef struct interlaced:
  double x
  double g
  long lag

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
def sagafast(A, double[:] b, props):
    
    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount
    cdef int indstart, indend, ylen, ind, yind, kp1
    cdef double cnew, activation, cchange, gscaling, ry, 
    cdef double yweight, gweight, cweight, tmp
    
    np.random.seed(42)
    
    cdef double gamma = props.get("stepSize", 0.1)
    cdef bool use_perm = props.get("usePerm", False)
    
    loss = getLoss(A, b, props)
    
    # Data points are stored in rows in CSR format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr
    
    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions
    
    cdef double[:] xk = np.zeros(m)
    cdef double[:] gk = np.zeros(m)
    # Tracks for each entry of x, what iteration it was last updated at.
    cdef unsigned int[:] lag = np.zeros(m, dtype='I')
    
    cdef interlaced[:] el = <interlaced[:m]>malloc(m*sizeof(interlaced))
    
    for ind in xrange(m):
      el[m].x = 0.0
      el[m].g = 0.0
      el[m].lag = 0
    
    cdef double reg = props.get('reg', 0.0001) 
    cdef double betak = 1.0 # Scaling factor for xk.
    
    logger = logging.getLogger("sagafast")

    cdef int maxiter = props.get("passes", 10)    
    
    # This is just a table of the sum the geometric series (1-reg*gamma)
    # It is used to correctly do the just-in-time updating when
    # L2 regularisation is used.
    cdef double[:] lag_scaling = np.zeros(n*maxiter+1)
    lag_scaling[0] = 0.0
    lag_scaling[1] = 1.0 - reg*gamma
    cdef double geosum = 1.0 - reg*gamma
    cdef double mult = 1.0 - reg*gamma
    for i in range(2,n+1):
        geosum *= mult
        lag_scaling[i] = lag_scaling[i-1] + geosum
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * betak * x product.
    cdef double[:] c = np.zeros(n)

    cdef unsigned int k = 0 # Current iteration number
    
    cdef long [:] ordering
    
    logger.info("sagafast starting, npoints=%d, ndims=%d" % (n, m))
    
    loss.store_training_loss(xk)
    
    for epoch in xrange(maxiter):
        if use_perm:
          if epoch % 2 == 0:
            ordering = np.arange(n, dtype='int64')
          else:
            ordering = np.random.permutation(n)
        else:
          if epoch == 0:
            ordering = np.arange(n, dtype='int64')
          else:
            ordering = np.random.random_integers(low=0, high=n-1, size=n)
          
        for j in xrange(n):
            i = ordering[j]
            
            # Selects the (sparse) column of the data matrix containing datapoint i.
            indstart = indptr[i]
            indend = indptr[i+1]
            ydata = data[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart
            
            # Apply the missed updates to xk just-in-time
            #lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling, -gamma/betak)
            gweight = -gamma/betak
            kp1 = k+1
            activation = 0.0
            tmp = 0.0
            
            for yind in xrange(0, ylen):
                ind = yindices[yind]
                lagged_amount = k-el[ind].lag
                el[ind].lag = kp1
                el[ind].x += lag_scaling[lagged_amount]*(gweight*el[ind].g)
          
                activation += ydata[yind]*el[ind].x
            
            #activation = betak * spdot(xk, ydata, yindices, ylen)
            activation = betak * activation # Correct for betak scaling
            
            cnew = loss.subgradient(i, activation)

            cchange = cnew-c[i]
            c[i] = cnew
            betak *= 1.0 - reg*gamma
            k += 1
            
            # Update xk with sparse step bit (with betak scaling)
            #add_weighted(xk, ydata, yindices, ylen, -cchange*gamma/betak)
            
            yweight = -cchange*gamma/betak
            gweight = -gamma/betak
            cweight = cchange/n
            for yind in xrange(ylen):
                ind = yindices[yind]
                el[ind].x += yweight*ydata[yind] + gweight*el[ind].g
                el[ind].g += cweight*ydata[yind]
                
            
            
            # Perform the gradient-average part of the step
            #lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling, -gamma/betak)
            
            # update the gradient average
            #add_weighted(gk, ydata, yindices, ylen, cchange/n) 
    
        logger.info("Epoch %d finished", epoch)
        
        # Unlag the vector
        gscaling = -gamma/betak
        for ind in xrange(m):
            lagged_amount = k-el[ind].lag
            el[ind].lag = k
            el[ind].x += lag_scaling[lagged_amount]*gscaling*el[ind].g
            el[ind].x = betak*el[ind].x
            xk[ind] = el[ind].x
        
        betak = 1.0
        
        if epoch % 20 == 0 and epoch > 0:
          loss.store_training_loss(xk)    
    
    logger.info("sagafast finished")
    free(&el[0])
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}


# Basic statistics on 50 passes of RCV1, reg 0.0001
# baseline: 2.3 to 2.37 user, as timed by "time" bash command
# Just removing the loss.store_training_loss: 1.91-2.03ish.
# After loop fusion before and after update: 1.71ish.
# After moving the random number gen into vectorized numy, 1.60ish
# Adding @cython.initializedcheck(False) @cython.nonecheck(False) @cython.overflowcheck(False) didn't really help. Still around 1.60.
# Switching to interlaced x/g/lag gives me 1.3s, a big improvement.
# Unrolling the first inner loop seemed to not help at all, even made things very slightly workse.
