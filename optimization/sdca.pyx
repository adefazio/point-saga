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
def sdca(A, double[:] b, props):
    
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
    
    cdef double reg = props.get('reg', 0.0001) 
    cdef bool use_perm = props.get("usePerm", False)
    
    loss = getLoss(A, b, props)
    
    cdef double lamb = 1.0/(reg*n)
    
    np.random.seed(42)
    
    logger = logging.getLogger("sdca")

    cdef int maxiter = props.get("passes", 10)    

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
        
          add_weighted(xk, ydata, yindices, ylen, -lamb*c[i])
    
    cdef unsigned int k = 0 # Current iteration number
    
    cdef long [:] perm = np.random.permutation(n)
    
    xlist = []
    flist = []
    errorlist = []
    
    logger.info("reg: %1.3e, lamb: %1.3e", reg, lamb)
    logger.info("SDCA starting, npoints=%d, ndims=%d", n, m)
    
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
            
            if epoch == 0:
              nseen = nseen + 1
              
        logger.info("Epoch %d finished", epoch)
        
        loss.store_training_loss(xk)    
    
    logger.info("SDCA finished")
    
    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
