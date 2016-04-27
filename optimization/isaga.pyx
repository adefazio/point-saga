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
    logger = logging.getLogger("isaga")
    
    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, ry, greg_old, reg_weight, tmp, Lp
    
    np.random.seed(42)
    
    cdef double regUpdatesPerPass = props.get("regUpdatesPerPass", 20)
    cdef bool use_perm = props.get("usePerm", False)
    
    # Data points are stored in rows in CSR format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr
    
    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions
    cdef double inv_n_plus_r = 1.0/(n + regUpdatesPerPass)

    cdef double reg = props.get('reg', 0.0001)
    cdef double[:] regs = reg*np.ones(m)
    
    cdef bool adaptive = props.get("adaptive", True)
      
    cdef double L = 1.0
    cdef double gamma = 1.0/(L+reg)
    
    if not adaptive:
      gamma = props.get("stepSize", 0.1)
    
    cdef bool normalize_data = props.get("normalizeData", True)
    
    cdef double[:] feature_weights = np.ones(m)
    cdef double[:] bdata, rdata, norm_sq
    
    
    if normalize_data:
      ## We will accumulate starting at reg value
      for i in range(m):
        feature_weights[i] = 0
      
      #logger.info("nentries? %d", indptr[n])
      # Compute column squared norms          
      for j in range(indptr[n]):
        feature_weights[indices[j]] += data[j]*data[j];
      
      #for i in range(m):
      #  feature_weights[i] = (4.0*n)/m
      
      # Calculate feature weights
      for i in range(m):
        if feature_weights[i] == 0:
          feature_weights[i] = 1.0
          regs[i] = reg
        else:
          tmp = sqrt(m * (reg + feature_weights[i]/n))
          feature_weights[i] = 1.0/tmp
          regs[i] = reg*feature_weights[i]*feature_weights[i]
        #logger.info("%d regs[i]: %1.2e,  fw[i]: %1.2e", i, regs[i], feature_weights[i])
      # Produce normalized copy of A data
      bdata = np.empty(indptr[n])
      
      for j in range(indptr[n]):
        bdata[j] = data[j]*feature_weights[indices[j]]
      
      # use bdata instead of A's data
      data = bdata
      
      logger.info("Percentiles of feature weights")
      perc = np.percentile(feature_weights, [0, 25, 50, 75, 100])
      logger.info(perc)
      
      norm_sq = np.zeros(n)
      for i in range(n):
        rdata = data[indptr[i]:indptr[i+1]]
        norm_sq[i] = np.dot(rdata, rdata)
      
      logger.info("Squared norm mean after renom: %1.7f", np.mean(norm_sq))
      logger.info("Squared norm percentiles [0, 25, 50, 75, 100] (After renorm):")
      perc = np.percentile(norm_sq, [0, 25, 50, 75, 100])
      logger.info(perc)
      logger.info("Max/mean: %1.2f", perc[4]/np.mean(norm_sq))
    
    loss = getLoss(A, b, props)
    cdef double[:] xk = np.zeros(m)
    cdef double[:] gk = np.zeros(m)
    cdef double[:] greg = np.zeros(m)
    
    cdef double[:] xkrenorm = np.zeros(m)
    
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
              
              reg_weight = (n/(regUpdatesPerPass+0.0))
              
              for ind in range(m):
                greg_old = greg[ind]
                greg[ind] = regs[ind]*reg_weight*xk[ind]
                # Main update for L2 norm term
                xk[ind] -= gamma*(greg[ind] - greg_old + gk[ind])
                gk[ind] += (greg[ind] - greg_old)*regUpdatesPerPass*inv_n_plus_r
            
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
            
            if adaptive:
              # Line search
              if True:
                Lp = loss.lipschitz(i, activation)
                if Lp < 1.1*L:
                  Lp = L
              else:
                Lp = loss.linesearch(i, activation, L)
              
              if Lp != L:
                flat_unlag(m, k, xk, gk, lag, -gamma)
                
                gamma = 1.0/(Lp+reg) 
                logger.info("Increasing L from %1.9f to %1.9f (gamma=%1.2e)", L, Lp, gamma)
                L = Lp
            
            # Update xk with sparse step bit
            add_weighted(xk, ydata, yindices, ylen, -cchange*gamma)
            
            k += 1
            
            # Perform the gradient-average part of the step
            flat_lagged_update(k, xk, gk, lag, yindices, ylen, -gamma)
            
            # update the gradient average
            add_weighted(gk, ydata, yindices, ylen, cchange/(n+regUpdatesPerPass)) 
    
        logger.info("Epoch %d finished", epoch)  
          
        flat_unlag(m, k, xk, gk, lag, -gamma)
        for i in range(m):
          xkrenorm[i] = xk[i]*feature_weights[i];
            
        loss.store_training_loss(xkrenorm)
        
        if adaptive:
          logger.info("L: %1.5f (halving for next pass)", L)
          L /= 2.0
          gamma = 1.0/(L+reg) 
    
    logger.info("isaga finished")
    
    for i in range(20):
      logger.info("xkr[%d]=%1.3e    (%1.3e)", i, xkrenorm[i], xk[i])
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
