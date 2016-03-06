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

from get_loss import getLoss

STUFF = "Hi"

ctypedef np.float32_t dtype_t

#cimport sparse_util
from sparse_util cimport spdot, add_weighted, lagged_update, add_weighted_diag, lagged_update_diag

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def diagonalsaga(A, double[:] b, props):
    
    # temporaries
    cdef double[:] ydata, ydata_h
    cdef int[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, ry
    cdef double hnew, hold, hchange
    
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
    
    cdef double reg = props.get('reg', 0.0001) 
    cdef double betak = 1.0 # Scaling factor for xk.
    
    logger = logging.getLogger("dsaga")

    cdef int maxiter = props.get("passes", 10)    
    
    # Tracks for each entry of x, what iteration it was last updated at.
    cdef unsigned int[:] lag = np.zeros(m, dtype='I')

    # This is just a table of the sum the geometric series (1-reg*gamma)
    # It is used to correctly do the just-in-time updating when
    # L2 regularisation is used.
    cdef double[:] lag_scaling = np.zeros(n*maxiter+1)
    lag_scaling[0] = 0.0
    lag_scaling[1] = 1.0
    cdef double geosum = 1.0
    cdef double mult = 1.0 - reg*gamma
    for i in range(2,n*maxiter+1):
        geosum *= mult
        lag_scaling[i] = lag_scaling[i-1] + geosum
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * betak * x product.
    cdef double[:] c = np.zeros(n)
    
    # For each data point, store it's hessian weighting
    cdef double[:] hweights = np.zeros(n)
    # Current diagonal estimate of the hessian
    cdef double[:] hk = reg*np.ones(m)
    
    # Compute square of every element of h for use in diagonal scaling.
    cdef double[:] data_h = np.zeros(len(data))
    
    for i in range(len(data)):
        data_h[i] = data[i]*data[i]
    
    # Initialize hk
    for i in range(n):
        indstart = indptr[i]
        indend = indptr[i+1]
        ydata_h = data_h[indstart:indend]
        yindices = indices[indstart:indend]
        ylen = indend-indstart
      
        hweights[i] = loss.hessianscale(i, 0.0)
        add_weighted(hk, ydata_h, yindices, ylen, hweights[i]/n)
    
    #Optional: Initialize gk
    if False:
      for i in range(n):
          indstart = indptr[i]
          indend = indptr[i+1]
          ydata = data[indstart:indend]
          yindices = indices[indstart:indend]
          ylen = indend-indstart
          ry = b[i]
        
          c[i] = loss.subgradient(i, 0.0)
        
          add_weighted(gk, ydata, yindices, ylen, c[i]/n)
    
    cdef unsigned int k = 0 # Current iteration number
    
    cdef long [:] perm = np.random.permutation(n)
    
    logger.info("diagonal-saga starting, npoints=%d, ndims=%d el=%d" % (n, m, len(data)))
    logger.info("hweights[0]=%1.4f, hk[0]=%1.4f", hweights[0], hk[0]) 
    
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
            ydata_h = data_h[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart
            
            # Apply the missed updates to xk just-in-time
            lagged_update_diag(k, xk, hk, gk, lag, yindices, ylen, lag_scaling, -gamma/betak)
            
            activation = betak * spdot(xk, ydata, yindices, ylen)
            
            cnew = loss.subgradient(i, activation)

            cchange = cnew-c[i]
            c[i] = cnew
            betak *= 1.0 - reg*gamma
            
            
            hnew = loss.hessianscale(i, activation)
            hold = hweights[i]
            
            hchange = hnew - hold
            hweights[i] = hnew
            
            #Update hk
            add_weighted(hk, ydata_h, yindices, ylen, hchange/n)
            
            # Update xk with sparse step bit (with betak scaling)
            add_weighted_diag(xk, hk, ydata, yindices, ylen, -cchange*gamma/betak)
            
            k += 1
            
            # Perform the gradient-average part of the step
            lagged_update_diag(k, xk, hk, gk, lag, yindices, ylen, lag_scaling, -gamma/betak)
            
            # update the gradient average
            add_weighted(gk, ydata, yindices, ylen, cchange/n) 
    
        logger.info("Epoch %d finished", epoch)
        
        # Unlag the vector
        gscaling = -gamma/betak
        for ind in range(m):
            lagged_amount = k-lag[ind]
            lag[ind] = k
            xk[ind] += lag_scaling[lagged_amount]*gscaling*gk[ind]
            xk[ind] = betak*xk[ind]
        
        betak = 1.0
        
        loss.store_training_loss(xk)    
    
    logger.info("diagonal-saga finished")
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
