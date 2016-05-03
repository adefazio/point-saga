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
@cython.boundscheck(False)
@cython.wraparound(False)
def flexsaga(A, double[:] b, props):
    
    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount, ls_update, r, most_visited
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, ry, Lk, Lkrev, Lprev
    cdef double yweight, gweight, cweight, tmp
    
    np.random.seed(42)
    
    loss = getLoss(A, b, props)
    
    # Data points are stored in rows in CSR format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr
    
    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions
    
    cdef double[:] xk = np.zeros(m)
    cdef double[:] gk = np.zeros(m)
    
    cdef interlaced[:] el = <interlaced[:m]>malloc(m*sizeof(interlaced))
    
    for ind in xrange(m):
      el[m].x = 0.0
      el[m].g = 0.0
      el[m].lag = 0
    
    cdef double reg = props.get('reg', 0.0001) 
    cdef double betak = 1.0 # Scaling factor for xk.
    
    #cdef double gamma = props.get("stepSize", 0.1)
    cdef double gamma_scale = props.get("gammaScale", 0.25)
    cdef double L = 1.0 + reg # Current Lipschitz used for step size
    cdef double Lavg = 1.0 + reg # Average Lipschitz across the points
    cdef double gamma = gamma_scale/L
    #cdef double L_shrink_factor = pow(2.0, -1.0/n)
    cdef bool use_perm = props.get("usePerm", False)
    
    logger = logging.getLogger("flexsaga")

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
    
    cdef double[:] ls = (1.0+reg)*np.ones(n)
    cdef unsigned int[:] visits = np.zeros(n, dtype='I')
    cdef unsigned int[:] visits_pass = np.zeros(n, dtype='I')
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * betak * x product.
    cdef double[:] c = np.zeros(n)
    
    if False:
      Lavg = 0.0
      for i in range(n):
          indstart = indptr[i]
          indend = indptr[i+1]
          ydata = data[indstart:indend]
          yindices = indices[indstart:indend]
          ylen = indend-indstart
          ry = b[i]
        
          #c[i] = loss.subgradient(i, 0.0)
          ls[i] = reg + loss.lipschitz(i, 0.0)
          Lavg += ls[i]/n
      
          #add_weighted(gk, ydata, yindices, ylen, c[i]/n)
      
      gamma = gamma_scale/Lavg
      L = Lavg
    
    cdef unsigned int k = 0 # Current iteration number
    
    ls_update = 2
    
    cdef FastSampler sampler = FastSampler(max_entries=n, max_value=100, min_value=1e-14)
    
    cdef bool use_seperate_update = props.get("useSeparateUpdate", True)
    
    cdef long [:] ordering
    
    logger.info("flexsaga starting, npoints=%d, ndims=%d, L=%1.1f, gammaScale=%1.3e" % (n, m, L, gamma_scale))
    
    loss.store_training_loss(xk)
    
    for epoch in range(maxiter):
            
        ordering = np.random.random_integers(low=0, high=n-1, size=n)
            
        for j in range(n):
            if epoch == 0:
                i = j
            else:
                i = sampler.sampleAndRemove()
                #visits_pass[i] += 1
                #visits[i] += 1
            
            # Selects the (sparse) column of the data matrix containing datapoint i.
            indstart = indptr[i]
            indend = indptr[i+1]
            ydata = data[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart
            
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
            
            activation = betak * activation # Correct for betak scaling
            
            cnew = loss.subgradient(i, activation)

            cchange = cnew-c[i]
            if epoch == 0 or not use_seperate_update:
              c[i] = cnew
            betak *= 1.0 - reg*gamma
            k += 1 
            
            # Line search
            Lprev =  ls[i]
            Lk = reg + loss.lipschitz(i, activation)
            
            Lavg += (Lk-ls[i])/n
            ls[i] = Lk
            
            # Add i back into sample pool with weight Lk
            sampler.add(i, Lk)
            
            #logger.info("Lavg: %1.9f Lk: %1.9f activation: %1.1e", Lavg, Lk, activation)
            if Lavg > 1.1*L:
              unlag(k, m, gamma, betak, lag, xk, gk, lag_scaling)
              betak = 1.0
              gamma = gamma_scale/Lavg 
              logger.info("Increasing L from %1.9f to %1.9f", L, Lavg)
              L = Lavg
              
              # Reset lag_scaling table for the new gamma
              mult = 1.0 - reg*gamma
              ls_update = 2
              geosum = 1.0
            
            yweight = -cchange*gamma*L/(Lprev*betak) #-cchange*gamma/betak
            gweight = -gamma/betak
            cweight = cchange/n
            for yind in xrange(ylen):
                ind = yindices[yind]
                el[ind].x += yweight*ydata[yind] + gweight*el[ind].g
            
            
            # use seperate sampling for the gradient table update?
            if epoch > 0 and use_seperate_update:
              # Uniform sampling for the gradient table update:
              r = ordering[j]
              indstart = indptr[r]
              indend = indptr[r+1]
              ydata = data[indstart:indend]
              yindices = indices[indstart:indend]
              ylen = indend-indstart
              
              gweight = -gamma/betak
              activation = 0.0
              tmp = 0.0
            
              for yind in xrange(0, ylen):
                  ind = yindices[yind]
                  lagged_amount = k-el[ind].lag
                  el[ind].lag = k
                  el[ind].x += lag_scaling[lagged_amount]*(gweight*el[ind].g)
          
                  activation += ydata[yind]*el[ind].x
            
              activation = betak * activation
              
              cnew = loss.subgradient(r, activation)

              # Table update
              cchange = cnew-c[r]
              c[r] = cnew
            
            cweight = cchange/n
            for yind in xrange(ylen):
                ind = yindices[yind]
                el[ind].g += cweight*ydata[yind] 
            
            # Update lag table
            geosum *= mult
            lag_scaling[ls_update] = lag_scaling[ls_update-1] + geosum
            ls_update += 1
                
        logger.info("Epoch %d finished", epoch)
        logger.info("L: %1.5f Lavg: %1.5f mean(ls): %1.5f", L, Lavg, np.mean(ls))
        
        # Unlag the vector
        gscaling = -gamma/betak
        for ind in xrange(m):
            lagged_amount = k-el[ind].lag
            el[ind].lag = k
            el[ind].x += lag_scaling[lagged_amount]*gscaling*el[ind].g
            el[ind].x = betak*el[ind].x
            xk[ind] = el[ind].x
        
        betak = 1.0
        
        #unlag(k, m, gamma, betak, lag, xk, gk, lag_scaling)
        #betak = 1.0
        if Lavg <= 0.5*L:
          logger.info("Decreasing, L: %1.5f Lavg: %1.5f", L, Lavg)
          #L /= 2.0 # Causes some instability
          L = 1.2*Lavg
          gamma = gamma_scale/L 
          
          mult = 1.0 - reg*gamma
          ls_update = 2
          geosum = 1.0
        
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef unlag(unsigned int k, unsigned int m, double gamma, double betak, 
          unsigned int[:] lag, double[:] xk, double[:] gk, double[:] lag_scaling):
  cdef unsigned int ind, lagged_amount
  
  # Unlag the vector
  cdef double gscaling = -gamma/betak
  for ind in range(m):
      lagged_amount = k-lag[ind]
      lag[ind] = k
      xk[ind] += lag_scaling[lagged_amount]*gscaling*gk[ind]
      xk[ind] = betak*xk[ind]
  
  betak = 1.0
