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
from sparse_util cimport spdot, add_weighted, flat_lagged_update, flat_unlag

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def isaga(A, double[:] b, props):
    
    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, ry, greg_old, reg_weight
    
    np.random.seed(42)
    
    cdef double gamma = props.get("stepSize", 0.1)
    cdef double regUpdatesPerPass = props.get("regUpdatesPerPass", 10)
    cdef bool use_perm = props.get("usePerm", False)
    
    cdef bool normalize_data = props.get("normalizeData", True)
    
    if normalize_data:
      # Compute column squared norms
      
      # Calculate feature weights
      
      # Produce normalized copy of A
      
      # Set regulariser weights
    
    loss = getLoss(A, b, props)
    
    # Data points are stored in rows in CSR format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr
    
    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions
    cdef double inv_n = 1.0/n
    
    cdef double[:] xk = np.zeros(m)
    cdef double[:] gk = np.zeros(m)
    cdef double[:] greg = np.zeros(m)
    
    cdef double reg = props.get('reg', 0.0001) 
    
    logger = logging.getLogger("isaga")

    cdef int maxiter = props.get("passes", 10)    
    
    # Tracks for each entry of x, what iteration it was last updated at.
    cdef unsigned int[:] lag = np.zeros(m, dtype='I')
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * x product.
    cdef double[:] c = np.zeros(n)
    
    if False:
      for i in range(n):
          indstart = indptr[i]
          indend = indptr[i+1]
          ydata = data[indstart:indend]
          yindices = indices[indstart:indend]
          ylen = indend-indstart
          ry = b[i]
        
          c[i] = loss.subgradient(i, 0.0)
        
          add_weighted(gk, ydata, yindices, ylen, c[i]/(n+regUpdatesPerPass))
    
    cdef unsigned int k = 0 # Current iteration number
    
    cdef long [:] perm = np.random.permutation(n)
    
    logger.info("isaga starting, npoints=%d, ndims=%d" % (n, m))
    
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
            
            # with probability regUpdatesPerPass/(n+regUpdatesPerPass) we want to sample the L2 regulariser instead. It has strength n*reg/regUpdatesPerPass each time.
            l2p = np.random.uniform()
            if l2p <= regUpdatesPerPass/(n+regUpdatesPerPass+0.0):
              #logger.info("Applying regulariser")
              # Unlag
              flat_unlag(m, k, xk, gk, lag, -gamma)
              
              reg_weight = (n*(reg/regUpdatesPerPass))
              
              for ind in range(m):
                greg_old = greg[ind]
                greg[ind] = reg_weight*xk[ind]
                # Main update for L2 norm term
                xk[ind] -= gamma*(greg[ind] - greg_old + gk[ind])
                gk[ind] += regUpdatesPerPass*inv_n*(greg[ind] - greg_old)
            
              continue
            
            # Selects the (sparse) column of the data matrix containing datapoint i.
            indstart = indptr[i]
            indend = indptr[i+1]
            ydata = data[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart
            
            # Apply the missed updates to xk just-in-time
            flat_lagged_update(k, xk, gk, lag, yindices, ylen, -gamma)
            
            activation = spdot(xk, ydata, yindices, ylen)
            
            cnew = loss.subgradient(i, activation)

            cchange = cnew-c[i]
            c[i] = cnew
            
            # Update xk with sparse step bit
            add_weighted(xk, ydata, yindices, ylen, -cchange*gamma)
            
            k += 1
            
            # Perform the gradient-average part of the step
            flat_lagged_update(k, xk, gk, lag, yindices, ylen, -gamma)
            
            # update the gradient average
            add_weighted(gk, ydata, yindices, ylen, cchange/(n+regUpdatesPerPass)) 
    
        logger.info("Epoch %d finished", epoch)
          
        flat_unlag(m, k, xk, gk, lag, -gamma)
        loss.store_training_loss(xk)    
    
    logger.info("isaga finished")
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
