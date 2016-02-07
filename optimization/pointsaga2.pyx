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

#from hinge_loss import HingeLoss
#from logistic_loss import LogisticLoss

#from loss import getLoss

STUFF = "Hi"

ctypedef np.float32_t dtype_t

#cimport sparse_util
from sparse_util cimport spdot, add_weighted, lagged_update
from get_loss import getLoss

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def pointsaga2(A, double[:] b, props):
    
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
    
    cdef double wscale = 1.0 # Scaling factor for xk.
    cdef double ry
    
    np.random.seed(42)
    
    logger = logging.getLogger("pointsaga2")

    cdef int maxiter = props.get("passes", 10)    
    
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
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * wscale * x product.
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
    
    cdef long [:] perm = np.random.permutation(n)
    
    xlist = []
    flist = []
    errorlist = []
    
    logger.info("Gamma: %1.2e, reg: %1.3e, prox_conversion_factor: %1.8f, 1-reg*gamma: %1.8f", 
                gamma, reg, prox_conversion_factor, 1.0 - reg*gamma)
    logger.info("Point-saga2 starting, npoints=%d, ndims=%d" % (n, m))
    
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
            
            if use_lag:
              lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling, -gamma/wscale)
              
              add_weighted(xk, ydata, yindices, ylen, c[i]*gamma/wscale)
              
              wscale *= prox_conversion_factor
            else:   
              #Apply gk (gradient average quantity)         
              for p in range(m):
                  xk[p] = xk[p] - gamma*gk[p]
              
              # Apply the old g_i
              add_weighted(xk, ydata, yindices, ylen, c[i]*gamma)

              # Scale xk
              for p in range(m):
                  xk[p] = prox_conversion_factor*xk[p]
            
            activation = wscale * spdot(xk, ydata, yindices, ylen)            
            # 
            #cnew = loss.prox(gamma_prime, i, activation)
            (new_loc, cnew) = loss.prox(gamma_prime, i, activation)
            
            cold = c[i]
            cchange = cnew-c[i]
            c[i] = cnew
            
            # Double check prox
            if True and loss.is_differentiable:
              sg = loss.subgradient(i, new_loc)
              #sg_atc = 
              if fabs(sg - cnew) > 1e-5:
                logger.info("Bad prox. sg: %1.8e, prox-sg: %1.8e, sg2: %1.8e, sg3: %1.8e",
                            sg, cnew, sg/gamma_prime, sg*gamma_prime)
            
            
            #logger.info("k=%d, ry=%2d, a=%1.3f, cold=%1.3f, cnew=%1.3f, new_loc=%1.3f, wscale=%1.1e, gamma: %1.1e, gamma_prime: %1.1e",
            #    k, ry, activation, cold, cnew, new_loc, wscale, gamma, gamma_prime)
            
            # Update xk with the f_j^prime(phi_j) (with wscale scaling)
            add_weighted(xk, ydata, yindices, ylen, -cnew*gamma_prime/wscale)
            
            # update the gradient average
            add_weighted(gk, ydata, yindices, ylen, cchange/n) 
            
            if epoch == 0:
              nseen = nseen + 1
              
        logger.info("Epoch %d finished", epoch)
        
        # Unlag the vector
        if use_lag:
          gscaling = -gamma/wscale
          for ind in range(m):
            lagged_amount = k-lag[ind]
            if lagged_amount > 0:
              lag[ind] = k
              xk[ind] += (lag_scaling[lagged_amount+1]-1)*gscaling*gk[ind]
            xk[ind] = wscale*xk[ind]
            
          wscale = 1.0
        
        loss.store_training_loss(xk)    
    
    logger.info("Point-saga2 finished")
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
