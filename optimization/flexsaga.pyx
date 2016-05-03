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

from fast_sampler cimport FastSampler

from get_loss import getLoss

STUFF = "Hi"

ctypedef np.float32_t dtype_t

#cimport sparse_util
from sparse_util cimport spdot, add_weighted, lagged_update, interlaced

@cython.cdivision(True)
@cython.boundscheck(True)
@cython.wraparound(False)
def flexsaga(A, double[:] b, props):
    
    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount, ls_update, r, most_visited
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, ry, Lk, Lkrev, Lprev
    cdef double yweight, gweight, cweight, tmp, reg_weight, greg_old
    
    logger = logging.getLogger("flexsaga")
    
    np.random.seed(42)
    
    loss = getLoss(A, b, props)
    
    # Data points are stored in rows in CSR format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr
    
    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions
    
    cdef double[:] xk = np.zeros(m)
    cdef double[:] greg = np.zeros(m)
    
    cdef interlaced[:] el = <interlaced[:m]>malloc(m*sizeof(interlaced))
    
   # logger.info("ind")
    for ind in xrange(m):
      el[ind].x = 0.0 # current iterate
      el[ind].g = 0.0 # gradient average quantity
      el[ind].lag = 0 # Tracks for each entry of x, what iteration it was last updated at.
    
    cdef int regTerms = props.get("regTerms", 20)
    cdef double reg = props.get('reg', 0.0001) 
    cdef double[:] regs = reg*np.ones(m)
    cdef double normalized_reg = reg
    #logger.info("a")
    
    cdef bool normalize_data = props.get("normalizeData", False)
    
    cdef double[:] feature_weights = np.ones(m)
    cdef double[:] bdata, rdata, norm_sq
    #logger.info("b")
    
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
    
    #logger.info("post norm")
    #cdef double gamma = props.get("stepSize", 0.1)
    cdef double gamma_scale = props.get("gammaScale", 0.15)
    cdef double L = 1.0 + reg # Current Lipschitz used for step size
    cdef double Lavg = 1.0 + reg # Average Lipschitz across the points
    cdef double gamma = gamma_scale/L
    
    cdef int maxiter = props.get("passes", 10)    
    
    cdef double[:] ls = (1.0+reg)*np.ones(n)
    cdef unsigned int[:] visits = np.zeros(n, dtype='I')
    cdef unsigned int[:] visits_pass = np.zeros(n, dtype='I')
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * betak * x product.
    cdef double[:] c = np.zeros(n)
    
    cdef unsigned int k = 0 # Current iteration number
    
    cdef FastSampler sampler = FastSampler(max_entries=n+1, max_value=100, min_value=1e-14)
    sampler.add(n, n*normalized_reg)
    
    cdef bool use_seperate_update = props.get("useSeparateUpdate", True)
    
    cdef long [:] ordering
    
    logger.info("flexsaga starting, npoints=%d, ndims=%d, L=%1.1f, gammaScale=%1.3e" % (n, m, L, gamma_scale))
    
    loss.store_training_loss(xk)
    
    for epoch in range(maxiter):
           
        if normalized_reg > 0: 
          ordering = np.random.random_integers(low=0, high=n+regTerms-1, size=n)
        else:
          ordering = np.random.random_integers(low=0, high=n-1, size=n)
            
        for j in range(n):
            if epoch == 0:
                i = j
                if normalized_reg > 0 and i % (n/regTerms) == 0 and i > 0:
                  i = n
            else:
                i = sampler.sampleAndRemove()
                #visits_pass[i] += 1
                #visits[i] += 1
            
            if i == n: # Regulariser term
              logger.info("Reg j=%d", j)
              Lk = n*normalized_reg
          
              for ind in range(m):
                lagged_amount = k-el[ind].lag
                el[ind].lag = k
                el[ind].x -= gamma*lagged_amount*el[ind].g
              
                greg_old = greg[ind]
                greg[ind] = n*regs[ind]*el[ind].x
                # Main update for L2 norm term
                el[ind].x -= (gamma*L/Lk)*(greg[ind] - greg_old)
                el[ind].x -= gamma*el[ind].g
                el[ind].g += (greg[ind] - greg_old)*regs[ind]
            
              # We don't want to miss any on the first pass
              if epoch == 0:
                i = j
              else:
                sampler.add(n, n*normalized_reg)
                continue
            
            # Selects the (sparse) column of the data matrix containing datapoint i.
            indstart = indptr[i]
            indend = indptr[i+1]
            ydata = data[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart
            #logger.info("a: normal j=%d i=%d", j, i)
            
            kp1 = k+1
            activation = 0.0
            
            for yind in xrange(0, ylen):
                ind = yindices[yind]
                lagged_amount = k-el[ind].lag
                el[ind].lag = kp1
                el[ind].x -= gamma*lagged_amount*el[ind].g
          
                activation += ydata[yind]*el[ind].x
            
            cnew = loss.subgradient(i, activation)

            cchange = cnew-c[i]
            if epoch == 0 or not use_seperate_update:
              c[i] = cnew
            k += 1 
            #logger.info("b: normal j=%d", j)
            
            # Update weight
            Lprev =  ls[i]
            Lk = reg + loss.lipschitz(i, activation)
            
            Lavg += (Lk-ls[i])/n
            ls[i] = Lk

            # Add i back into sample pool with weight Lk
            # TODO: maybe have a minimum value for this for numerical stability
            sampler.add(i, Lk)
            #logger.info("c: normal j=%d", j)
            #logger.info("Lavg: %1.9f Lk: %1.9f activation: %1.1e", Lavg, Lk, activation)
            if Lavg > 1.1*L:
              # Unlag
              #logger.info("Unlag")
              for ind in xrange(m):
                  lagged_amount = k-el[ind].lag
                  el[ind].lag = k
                  el[ind].x -= gamma*lagged_amount*el[ind].g
              
              gamma = gamma_scale/Lavg 
              logger.info("Increasing L from %1.9f to %1.9f", L, Lavg)
              L = Lavg
              
              # Reset lag_scaling table for the new gamma
              mult = 1.0 - reg*gamma
              ls_update = 2
              geosum = 1.0
            
            yweight = -cchange*gamma*L/Lprev #-cchange*gamma
            gweight = -gamma
            cweight = cchange/n
            for yind in xrange(ylen):
                ind = yindices[yind]
                el[ind].x += yweight*ydata[yind] + gweight*el[ind].g
            
            
            # use seperate sampling for the gradient table update?
            if epoch > 0 and use_seperate_update:
              # Uniform sampling for the gradient table update:
              r = ordering[j]
              if r >= n: # Regulariser terms
                for ind in range(m):
                  # Unlag everything first
                  lagged_amount = k-el[ind].lag
                  el[ind].lag = k
                  el[ind].x -= gamma*lagged_amount*el[ind].g
                  
                  greg_old = greg[ind]
                  greg[ind] = n*regs[ind]*el[ind].x
                  el[ind].g += (greg[ind] - greg_old)*regs[ind]
            
                continue
              else:
                indstart = indptr[r]
                indend = indptr[r+1]
                ydata = data[indstart:indend]
                yindices = indices[indstart:indend]
                ylen = indend-indstart
              
                activation = 0.0
            
                for yind in xrange(0, ylen):
                    ind = yindices[yind]
                    lagged_amount = k-el[ind].lag
                    el[ind].lag = k
                    el[ind].x -= gamma*lagged_amount*el[ind].g
          
                    activation += ydata[yind]*el[ind].x
              
                cnew = loss.subgradient(r, activation)

                # Table update
                cchange = cnew-c[r]
                c[r] = cnew
            
            cweight = cchange/n
            for yind in xrange(ylen):
                ind = yindices[yind]
                el[ind].g += cweight*ydata[yind] 
                
        logger.info("Epoch %d finished", epoch)
        logger.info("L: %1.5f Lavg: %1.5f mean(ls): %1.5f", L, Lavg, np.mean(ls))
        
        # Unlag the vector
        for ind in xrange(m):
            lagged_amount = k-el[ind].lag
            el[ind].lag = k
            el[ind].x -= gamma*lagged_amount*el[ind].g
            xk[ind] = el[ind].x*feature_weights[ind]
        
        if Lavg <= 0.5*L:
          logger.info("L: %1.5f Lavg: %1.5f, new L: %1.5f", L, Lavg, 1.2*Lavg)
          #L /= 2.0 # Causes some instability
          L = 1.2*Lavg
          gamma = gamma_scale/L
        
        loss.store_training_loss(xk)    
        
        # Some diagnostics on the sampler
        # reset pass counter
        #visits_pass[:] = 0
    
    
    logger.info("Sampler weight mean: %1.6f", np.mean(sampler.weights))
    logger.info("Sampler weights quantiles [0%, 25%, 50%, 75%, 100%]:")
    perc = np.percentile(sampler.weights, [0, 25, 50, 75, 100])
    logger.info(perc)
    #logger.info("Visit count quantiles:")
    #logger.info(np.percentile(visits, [0, 25, 50, 75, 100]))
    #most_visited = np.argmax(visits)
    #logger.info("Most visited: %d, visits %d, weight %1.2e", most_visited, visits[most_visited], sampler.weights[most_visited])
    
    logger.info("flexsaga finished")
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
