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
from sparse_util cimport spdot, add_weighted, lagged_update

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def wsaga(A, double[:] b, props):
    
    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount, ls_update, r
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, ry, Lp
    
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
    
    cdef double reg = props.get('reg', 0.0001) 
    cdef double betak = 1.0 # Scaling factor for xk.
    
    #cdef double gamma = props.get("stepSize", 0.1)
    cdef double L = 1.0 # Current Lipschitz used for step size
    cdef double Lavg = 1.0 # Average Lipschitz across the points
    cdef double gamma = 0.34/(L+reg)
    #cdef double L_shrink_factor = pow(2.0, -1.0/n)
    cdef bool use_perm = props.get("usePerm", False)
    
    logger = logging.getLogger("wsaga")

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
    
    cdef double[:] ls = np.ones(n)
    cdef unsigned int[:] visits = np.zeros(n, dtype='I')
    cdef unsigned int[:] visits_pass = np.zeros(n, dtype='I')
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * betak * x product.
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
        
          add_weighted(gk, ydata, yindices, ylen, c[i]/n)
    
    cdef unsigned int k = 0 # Current iteration number
    
    ls_update = 2
    
    cdef FastSampler sampler = FastSampler(max_entries=n, max_value=100, min_value=1e-14)
    
    cdef bool use_seperate_update = props.get("useSeparateUpdate", True)
    
    logger.info("WSaga starting, npoints=%d, ndims=%d, L=%1.1f" % (n, m, L))
    
    loss.store_training_loss(xk)
    
    for epoch in range(maxiter):
            
        for j in range(n):
            if epoch == 0:
                i = j 
            else:
                i = sampler.sampleAndRemove()
                visits_pass[i] += 1
                visits[i] += 1
            
            # Selects the (sparse) column of the data matrix containing datapoint i.
            indstart = indptr[i]
            indend = indptr[i+1]
            ydata = data[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart
            
            # Apply the missed updates to xk just-in-time
            lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling, -gamma/betak)
            
            activation = betak * spdot(xk, ydata, yindices, ylen)
            
            cnew = loss.subgradient(i, activation)

            cchange = cnew-c[i]
            if not use_seperate_update:
              c[i] = cnew
            betak *= 1.0 - reg*gamma
            
            # Line search
            if True:
              Lp = loss.lipschitz(i, activation)
              
              if True:
                Lavg += (Lp-ls[i])/n
                ls[i] = Lp
              else:
                Lavg = Lp
            
            # Add i back into sample pool with weight Lp
            #if Lp < reg:
            #  logger.info("Lp: %1.14f, activation: %1.14f, i:%d", Lp, activation, i)
            sampler.add(i, Lp)
              
            if Lavg > 1.1*L:
              unlag(k, m, gamma, betak, lag, xk, gk, lag_scaling)
              betak = 1.0
              gamma = 0.34/(Lavg+reg) 
              logger.info("Increasing L from %1.9f to %1.9f", L, Lavg)
              L = Lavg
              
              # Reset lag_scaling table for the new gamma
              mult = 1.0 - reg*gamma
              ls_update = 2
              geosum = 1.0
            
            # Update xk with sparse step bit (with betak scaling)
            add_weighted(xk, ydata, yindices, ylen, -cchange*gamma/betak)
            
            k += 1
            
            # Perform the gradient-average part of the step
            lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling, -gamma/betak)
            
            
            # use seperate sampling for the gradient table update?
            if use_seperate_update:
              # Uniform sampling for the gradient table update:
              r = np.random.randint(0, n)
              indstart = indptr[r]
              indend = indptr[r+1]
              ydata = data[indstart:indend]
              yindices = indices[indstart:indend]
              ylen = indend-indstart
              
              # Apply the missed updates to xk just-in-time
              lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling, -gamma/betak)
          
              activation = betak * spdot(xk, ydata, yindices, ylen)
          
              cnew = loss.subgradient(r, activation)

              # Table update
              cchange = cnew-c[r]
              c[r] = cnew
              
              # Update gradient average for uniformly sampled point.
              add_weighted(gk, ydata, yindices, ylen, cchange/n) 
            else:
              # Update gradient average for non-uniformly sampled point.
              add_weighted(gk, ydata, yindices, ylen, cchange/n) 
            
            # Update lag table
            geosum *= mult
            lag_scaling[ls_update] = lag_scaling[ls_update-1] + geosum
            ls_update += 1
                
        logger.info("Epoch %d finished", epoch)
        logger.info("L: %1.5f Lavg: %1.5f", L, Lavg)
        
        unlag(k, m, gamma, betak, lag, xk, gk, lag_scaling)
        betak = 1.0
        L /= 2.0
        gamma = 1.0/(L+reg) 
        
        ls_update = 2
        geosum = 1.0
        
        loss.store_training_loss(xk)    
        
        # Some diagnostics on the sampler
        #TODO
        # reset pass counter
        visits_pass[:] = 0
    
    
    logger.info("Sampler weight mean: %1.6f", np.mean(sampler.weights))
    logger.info("Sampler weights quantiles [0%, 25%, 50%, 75%, 100%]:")
    perc = np.percentile(sampler.weights, [0, 25, 50, 75, 100])
    logger.info(perc)
    logger.info("Visit count quantiles:")
    logger.info(np.percentile(visits, [0, 25, 50, 75, 100]))
    most_visited = np.argmax(visits)
    logger.info("Most visited: %d, visits %d, weight %1.2e", most_visited, visits[most_visited], sampler.weights[most_visited])
    
    logger.info("W-Point-saga finished")
    
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
