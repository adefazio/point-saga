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

STUFF = "Hi"

ctypedef np.float32_t dtype_t

# Performs the lagged update of x by g.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void lagged_update(long k, double[:] x, double[:] g, unsigned int[:] lag, 
                          int[:] yindices, int ylen, double[:] lag_scaling, double a):
    
    cdef unsigned int i
    cdef unsigned int ind
    cdef unsigned long lagged_amount = 0
    
    for i in range(ylen):
        ind = yindices[i]
        lagged_amount = k-lag[ind]
        lag[ind] = k
        x[ind] += lag_scaling[lagged_amount]*(a*g[ind])

# Performs the lagged update of x by g.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void lagged_update_with_xbar(long k, double[:] x, double[:] xbar, double alpha_k, 
                          double[:] g, unsigned int[:] lag, 
                          int[:] yindices, int ylen, double[:] lag_scaling, double a):

    cdef unsigned int i
    cdef unsigned int ind
    cdef unsigned long lagged_amount = 0

    for i in range(ylen):
        ind = yindices[i]
        lagged_amount = k-lag[ind]
        lag[ind] = k
        x[ind] += lag_scaling[lagged_amount]*(a*g[ind])  
        xbar[ind] -= alpha_k*lag_scaling[lagged_amount]*(a*g[ind])  

# Performs x += a*y, where x is dense and y is sparse.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add_weighted(double[:] x, double[:] ydata , int[:] yindices, int ylen, double a):
    cdef unsigned int i
    
    for i in range(ylen):
        #print "i: %d" % i
        #print "ylen: %d" % ylen
        #print "ydata: %1.1e" % ydata[i]
        x[yindices[i]] += a*ydata[i]

# Dot product of a dense vector with a sparse vector
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double spdot(double[:] x, double[:] ydata , int[:] yindices, int ylen):
    cdef unsigned int i
    cdef double v = 0.0
    
    for i in range(ylen):
        v += ydata[i]*x[yindices[i]]
        
    return v


######################

# Performs the lagged update of x by g.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void lagged_update_diag(
  long k, double[:] x, double[:] hk, double[:] g, unsigned int[:] lag, 
  int[:] yindices, int ylen, double[:] lag_scaling, double a):
    
    cdef unsigned int i
    cdef unsigned int ind
    cdef unsigned long lagged_amount = 0
    
    for i in range(ylen):
        ind = yindices[i]
        lagged_amount = k-lag[ind]
        lag[ind] = k
        x[ind] += lag_scaling[lagged_amount]*(a*g[ind])/sqrt(hk[ind])

# Performs x += a*y, where x is dense and y is sparse.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add_weighted_diag(double[:] x, double[:] hk, double[:] ydata , 
                            int[:] yindices, int ylen, double a):
    cdef unsigned int i
    cdef unsigned int ind

    for i in range(ylen):
        ind = yindices[i]
        x[ind] += a*ydata[i]/sqrt(hk[ind])
