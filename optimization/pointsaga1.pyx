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
from sparse_util cimport spdot, add_weighted, lagged_update, lagged_update_with_xbar
from get_loss import getLoss

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def pointsaga1(A, double[:] b, props):
    
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
    
    cdef double[:] xk = np.zeros(m)
    cdef double[:] gk = np.zeros(m)
    
    cdef double gamma = props.get("stepSize", 0.1)
    cdef double reg = props.get('reg', 0.0001) 
    cdef bool use_lag = props.get("useLag", True)
    cdef bool use_perm = props.get("usePerm", False)
    
    cdef int averageAfter = props.get("averageAfter", n)
    cdef bool improved_averaging =  props.get("improvedAveraging", True)
    cdef bool return_averaged = props.get("returnAveraged", False)
    
    loss = getLoss(A, b, props)
    
    cdef double wscale = 1.0 # Scaling factor for xk.
    cdef double ry, mu
    
    cdef double [:] wbar = np.zeros(m)
    cdef double [:] wbar_test = np.zeros(m)
    cdef double [:] wbar_interm = np.zeros(m)
    
    cdef double alpha_k = 0
    cdef double beta_k = 1
    cdef unsigned int t = 0
    
    np.random.seed(42)
    
    logger = logging.getLogger("pointsaga1")

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
    lag_scaling[1] = 1.0
    cdef double geosum = 1.0
    cdef double mult = prox_conversion_factor #TODO: not sure about this line
    for i in range(2,n+2):
        geosum *= mult
        lag_scaling[i] = lag_scaling[i-1] + geosum
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * wscale * x product.
    cdef double[:] c = np.zeros(n)
    
    cdef unsigned int k = 0 # Current iteration number
    
    cdef long [:] perm = np.random.permutation(n)
    
    xlist = []
    flist = []
    errorlist = []
    
    logger.info("Gamma: %1.2e, reg: %1.3e, prox_conversion_factor: %1.8f, 1-reg*gamma: %1.8f", 
                gamma, reg, prox_conversion_factor, 1.0 - reg*gamma)
    logger.info("Point-saga1 starting, npoints=%d, ndims=%d" % (n, m))
    
    loss.store_training_loss(xk)   
    
    for epoch in range(maxiter):
            
        for j in range(n):
            t = t + 1
            
            if epoch == 0:
                i = perm[j]
            else:
              if use_perm:
                if epoch % 2 == 0:
                  i = perm[j]
                else:
                  i = j
              else:
                i = np.random.randint(0, n)
            
            if t >= averageAfter:
              if improved_averaging:
                mu = 2.0/(t+2)
              else:
                mu = 1.0/(t+1) # Averaging constant.
            else:
              mu = 1.0 
              alpha_k = 0
              beta_k = 1
              
            #logger.info("t: %d, averageAfter: %d, alpha_k: %1.5e, beta_k: %1.5e, mu: %1.5e wscale: %1.10e",
            #             t, averageAfter, alpha_k, beta_k, mu, wscale)
            
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
              
              lagged_update_with_xbar(k, xk, wbar_interm, alpha_k/prox_conversion_factor, gk, lag, yindices, ylen, lag_scaling, -gamma/wscale)
              
              add_weighted(xk, ydata, yindices, ylen, c[i]*gamma/wscale)
              add_weighted(wbar_interm, ydata, yindices, ylen, -(alpha_k*c[i]*gamma/(prox_conversion_factor*wscale)))
            
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
            
            # Update xk with the f_j^prime(phi_j) (with wscale scaling)
            add_weighted(xk, ydata, yindices, ylen, -cnew*gamma_prime/wscale)
            add_weighted(wbar_interm, ydata, yindices, ylen, alpha_k*cnew*gamma_prime/wscale)
              
            # update the gradient average          
            add_weighted(gk, ydata, yindices, ylen, cchange/n)
              
            if not use_lag:
              for p in range(m):
                  #wbar_test[p] = wbar_test[p] + mu*(wscale*w[p] - wbar_test[p])
                  wbar[p] = wbar[p] + mu*(wscale*xk[p] - wbar[p])
              
            if mu >= 1:
              alpha_k = wscale
              beta_k = 1
            else:
              beta_k = beta_k/(1.0-mu)
              alpha_k = alpha_k + mu*beta_k*wscale
              
        logger.info("Epoch %d finished", epoch)
        
        
        if use_lag:
          # Unlag the vector
          gscaling = -gamma/wscale
          for ind in range(m):
            lagged_amount = k-lag[ind]
            if lagged_amount > 0:
              lag[ind] = k
              xk[ind] += (lag_scaling[lagged_amount+1]-1)*gscaling*gk[ind]
              wbar_interm[ind] -= alpha_k*(lag_scaling[lagged_amount+1]-1)*gscaling*gk[ind]
          
          # Unscale wbar and xk
          for p in range(m):
              wbar[p] = (wbar_interm[p] + alpha_k*xk[p])/beta_k
              wbar_interm[p] = wbar[p]
              xk[p] = wscale*xk[p]
        
          alpha_k = 0 # after wscale changed.
          beta_k = 1
          wscale = 1
        
        if return_averaged:
          loss.store_training_loss(wbar, "wbar:")
          loss.compute_loss(xk, "x:")
        else:
          loss.store_training_loss(xk, "x:")
          
    logger.info("Point-saga1 finished")
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
