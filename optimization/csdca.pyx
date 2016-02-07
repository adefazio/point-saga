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
def csdca(A, double[:] b, props):
    
    # temporaries
    cdef double[:] ydata
    cdef int[:] yindices
    cdef unsigned int i, j, p, epoch, outerepoch, lagged_amount
    cdef int indstart, indend, ylen, ind
    cdef double cnew, activation, cchange, gscaling, cold, sg, new_loc
    cdef double alpha_new
    
    # Data points are stored in columns in CSC format.
    cdef double[:] data = A.data
    cdef int[:] indices = A.indices
    cdef int[:] indptr = A.indptr
    
    cdef unsigned int n = A.shape[0] # datapoints
    cdef unsigned int m = A.shape[1] # dimensions
    
    cdef double[:] xk = np.zeros(m)
    cdef double[:] xk_old = np.zeros(m)
    cdef double[:] yk = np.zeros(m)
    cdef double[:] yk_old = np.zeros(m)
    cdef double[:] yk_prime = np.zeros(m)
    cdef double[:] gk = np.zeros(m)
    
    cdef double kappa = props.get("stepSize", 0.1)
    cdef double reg = props.get('reg', 0.0001) 
    cdef bool use_lag = props.get("useLag", True)
    cdef bool use_perm = props.get("usePerm", False)
    
    cdef double reg_conversion_factor = 1-(reg/kappa)/(1+reg/kappa)
    
    cdef double lamb = reg_conversion_factor/(n*kappa)
    cdef double q = reg/(reg+kappa)
    cdef double alpha = sqrt(q)
    cdef double beta = alpha*(1-alpha)/(alpha*alpha + alpha)
    
    loss = getLoss(A, b, props)
    
    cdef double wscale = 1.0 # Scaling factor for xk.
    cdef double ry, mu
    
    np.random.seed(42)
    
    logger = logging.getLogger("csdca")

    cdef int maxiter = props.get("passes", 10)    
    cdef int maxinner = props.get("maxinner", 1)
    
    # For linear learners, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the activation * wscale * x product.
    cdef double[:] c = np.zeros(n)
    
    cdef unsigned int k = 0 # Current iteration number
    
    cdef long [:] perm = np.random.permutation(n)
    
    xlist = []
    flist = []
    errorlist = []
    
    logger.info("reg: %1.3e, lamb: %1.3e, q: %1.3e", reg, lamb, q)
    logger.info("CSDCA starting, npoints=%d, ndims=%d" % (n, m))
    
    logger.info("alpha=%1.3e, beta=%1.3e, reg_conversion_factor=%1.7e", alpha, beta, reg_conversion_factor)
    
    loss.store_training_loss(xk)   
    
    for outerepoch in range(maxiter):
      for epoch in range(maxinner):
            
          for j in range(n):
            
              if epoch == 0 and outerepoch == 0:
                  i = perm[j]
              else:
                  i = np.random.randint(0, n)
            
              indstart = indptr[i]
              indend = indptr[i+1]
              ydata = data[indstart:indend]
              yindices = indices[indstart:indend]
              ylen = indend-indstart
              ry = b[i]
            
              k += 1
            
              # Remove old g_j from xk
              add_weighted(xk, ydata, yindices, ylen, lamb*c[i])
            
              activation = spdot(xk, ydata, yindices, ylen)            
            
              (new_loc, cnew) = loss.prox(lamb, i, activation)
            
              cold = c[i]
              cchange = cnew-c[i]
              c[i] = cnew
            
              # Double check prox
              if True and loss.is_differentiable:
                sg = loss.subgradient(i, new_loc)
                if fabs(sg - cnew) > 1e-5:
                  logger.info("Bad prox. sg: %1.8e, prox-sg: %1.8e", sg, cnew)
            
              # Put g_j back into xk
              add_weighted(xk, ydata, yindices, ylen, -lamb*cnew)
        
          logger.info("CSDCA Inner epoch finished")
      
      asq = alpha*alpha 
      alpha_new = 0.5*(sqrt(q*q - 2*q*asq + asq*asq + 4*asq) + q - asq)
      beta = alpha*(1-alpha)/(asq + alpha_new)
      
      #logger.info("CSDCA Outer. alpha=%1.3e, alpha_new=%1.3e, beta=%1.3e", alpha, alpha_new, beta)
      
      for p in range(m):
        yk[p] = xk[p] + beta*(xk[p] - xk_old[p])
        xk_old[p] = xk[p]
      
      alpha = alpha_new
      
      # Set new xk
      for p in range(m):
        yk[p] = reg_conversion_factor*yk[p]
        xk[p] = xk[p] - yk_old[p] + yk[p]
        yk_old[p] = yk[p]
      
      #loss.compute_loss(xk, "xk:") 
      loss.store_training_loss(xk_old, "xk:")
        
    logger.info("CSDCA finished")
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
