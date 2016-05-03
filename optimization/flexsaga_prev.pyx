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

from fast_sampler cimport FastSampler
from get_loss import getLoss

STUFF = "Hi"

ctypedef np.float32_t dtype_t

#cimport sparse_util
from sparse_util cimport spdot, add_weighted, flat_lagged_update, flat_unlag

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def flexsaga(A, double[:] b, props):
    logger = logging.getLogger("flexsaga")
    
    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, ry, greg_old, reg_weight, 
    cdef double tmp, stepweight
    
    cdef int r, indstart_r, indend_r, ylen_r
    cdef double[:] ydata_r
    cdef int[:] yindices_r
    cdef double activation_r
    
    np.random.seed(42)
    
    cdef double regUpdatesFirstPass = props.get("regUpdatesFirstPass", 20)
    
    # Data points are stored in rows in CSR format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr
    
    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions
    cdef double inv_n = 1.0/n

    cdef double reg = props.get('reg', 0.0001)
    cdef double[:] regs = reg*np.ones(m)
    
    cdef bool normalize_data = props.get("normalizeData", False)
    
    cdef double[:] feature_weights = np.ones(m)
    cdef double[:] bdata, rdata, norm_sq
    cdef double normalized_reg = reg
    
    if normalize_data:
      logger.info("Normalizing data beforehand")
      ## We will accumulate starting at reg value
      for i in range(m):
        feature_weights[i] = 0
        
      # Compute column squared norms          
      for j in range(indptr[n]):
        feature_weights[indices[j]] += data[j]*data[j];
      
      # Calculate feature weights
      for i in range(m):
        if feature_weights[i] == 0:
          feature_weights[i] = 1.0
          regs[i] = reg
        else:
          tmp = sqrt(m * (reg + feature_weights[i]/n))
          feature_weights[i] = 1.0/tmp
          regs[i] = reg*feature_weights[i]*feature_weights[i]
          normalized_reg = 0
      
      # Produce normalized copy of A data
      bdata = np.empty(indptr[n])
      
      for j in range(indptr[n]):
        bdata[j] = data[j]*feature_weights[indices[j]]
      
      # use bdata instead of A's data
      data = bdata
      
    cdef FastSampler sampler = FastSampler(max_entries=n+1, max_value=1000, min_value=1e-6)
    # Add regulariser to sampler
    sampler.add(n, n*normalized_reg)
    
    logger.info("Percentiles of feature weights")
    perc = np.percentile(feature_weights, [0, 25, 50, 75, 100])
    logger.info(perc)
    
    
    cdef double gamma_scale = props.get("gammaScale", 0.1)
    cdef double Lavg = reg # Average Lipschitz across the points
    cdef double[:] ls = np.ones(n) # L for each
    
    norm_sq = np.zeros(n)
    for i in range(n):
      rdata = data[indptr[i]:indptr[i+1]]
      norm_sq[i] = np.dot(rdata, rdata)
      Lk = 0.25*norm_sq[i]
      sampler.add(i, Lk)
      ls[i] = Lk
      Lavg += ls[i]*inv_n
    
    cdef double L = Lavg
    cdef double gamma = gamma_scale/L
    
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
    
    cdef unsigned int k = 0 # Current (non-reg) iteration number
    
    logger.info("flexsaga starting, npoints=%d, ndims=%d, L=%1.5f", n, m, L)
    
    loss.store_training_loss(xk)
    
    for epoch in range(maxiter):
            
        for j in range(n):
            #i = sampler.sampleAndRemove()
            i = np.random.randint(0, n)
            
            # Regulariser
            if i == n:
              logger.info("Reg update")
              flat_unlag(m, k, xk, gk, lag, -gamma)
              
              #reg_weight = (n/(regUpdatesFirstPass+0.0))
              sampler.add(n, n*normalized_reg)
              #-cchange*gamma*L/Lprev
              if epoch == 0:
                stepweight = gamma*(n/(regUpdatesFirstPass+0.0))
              else:
                stepweight = gamma*L/(n*normalized_reg)
              
              for ind in range(m):
                greg_old = greg[ind]
                greg[ind] = n*regs[ind]*xk[ind]
                # Main update for L2 norm term
                xk[ind] -= stepweight*(greg[ind] - greg_old) + gamma*gk[ind]
                gk[ind] += (greg[ind] - greg_old)*inv_n
              
              continue
            
            # Selects the (sparse) column of the data matrix containing datapoint i.
            indstart = indptr[i]
            indend = indptr[i+1]
            ydata = data[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart
            
            if True:
              # Uniform sampling for the gradient table update:
              r = np.random.randint(0, n)
              indstart_r = indptr[r]
              indend_r = indptr[r+1]
              ydata_r = data[indstart_r:indend_r]
              yindices_r = indices[indstart_r:indend_r]
              ylen_r = indend_r-indstart_r
          
              # Apply the missed updates to xk just-in-time
              flat_lagged_update(k, xk, gk, lag, yindices_r, ylen_r, -gamma)
              
              activation_r = spdot(xk, ydata_r, yindices_r, ylen_r)
            
            # Apply the missed updates to xk just-in-time
            flat_lagged_update(k, xk, gk, lag, yindices, ylen, -gamma)
            
            activation = spdot(xk, ydata, yindices, ylen)
            
            cnew = loss.subgradient(i, activation)

            cchange = cnew-c[i]
            c[i] = cnew
            
            # Update the table of Lipschitz values
            Lprev = ls[i]
            #Lk = loss.lipschitz(i, activation)
            Lk = 0.5
            
            Lavg += (Lk-ls[i])*inv_n
            ls[i] = Lk
            
            # Add i back into sample pool with weight Lk
            #sampler.add(i, Lk)
            
            if Lavg > 1.1*L: #TODO: maybe move after add_weighted?
              flat_unlag(m, k, xk, gk, lag, -gamma)
              #unlag(k, m, gamma, betak, lag, xk, gk, lag_scaling)
              gamma = gamma_scale/Lavg #gamma = 1.0/(Lp+reg)?? 
              logger.info("Increasing L from %1.9f to %1.9f", L, Lavg)
              L = Lavg
            
            # Update xk with sparse step bit
            add_weighted(xk, ydata, yindices, ylen, -cchange*gamma*L/Lprev)
            
            k += 1
            
            # Perform the gradient-average part of the step
            flat_lagged_update(k, xk, gk, lag, yindices, ylen, -gamma)
            #flat_single_step(k, xk, gk, lag, yindices, ylen, -gamma)
            
            ##################
            # update the gradient average
            if True:
              cnew = loss.subgradient(r, activation_r)

              # Table update
              cchange = cnew-c[r]
              c[r] = cnew
            
            # Update gradient average for uniformly sampled point.
            add_weighted(gk, ydata, yindices, ylen, cchange*inv_n)
    
        logger.info("Epoch %d finished", epoch)  
          
        flat_unlag(m, k, xk, gk, lag, -gamma)
        for i in range(m):
          xkrenorm[i] = xk[i]*feature_weights[i];
            
        loss.store_training_loss(xkrenorm)
        
        logger.info("L: %1.5f (Lavg:%1.4f), mean(ls): %1.4f", L, Lavg, np.mean(ls))
        L = Lavg
        gamma = gamma_scale/L 
    
    logger.info("flexsaga finished")
    
    for i in range(20):
      logger.info("xkr[%d]=%1.3e    (%1.3e)", i, xkrenorm[i], xk[i])
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
