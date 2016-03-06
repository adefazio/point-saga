#cython: profile=False
import logging

import numpy as np
cimport numpy as np

cimport loss
from loss cimport Loss

cimport cython

cdef class SquaredLoss(Loss):

  def __init__(self, Xn, double [:] dn, props):
    super(SquaredLoss, self).__init__(Xn, dn, props)
    self.is_differentiable = True
   
  @cython.cdivision(True)
  @cython.boundscheck(False) 
  cpdef double subgradient(self, unsigned int i, double activation):
    cdef double ry = self.d[i]
     
    return(activation - ry)

  @cython.cdivision(True)
  @cython.boundscheck(False) 
  cpdef double hessianscale(self, unsigned int i, double activation):
    return(1.0)

  @cython.cdivision(True)
  @cython.boundscheck(False)
  cpdef tuple prox(self, double gamma, unsigned int i, double activation):
    cdef double sg, xp_old
    cdef double gamma_scaled = gamma*self.norm_sq[i]
    cdef double ry = self.d[i]
    
    xp = (activation + gamma_scaled*ry)/(gamma_scaled+1.0)
  
    sg = (activation - xp)/gamma_scaled

    return((xp, sg))
      
    
  cpdef compute_loss(self, double[:] w, txt=""):
    
    logger = logging.getLogger("squaredloss")
    
    cdef double [:] rdata
    cdef int [:] rind
    cdef double rw, ry
    cdef int p,i
    
    cdef double loss = 0.0
    cdef int errors = 0
    
    for i in range(self.npoints):
        rlen = self.indptr[i+1]-self.indptr[i]
        rind = self.indices[self.indptr[i]:self.indptr[i+1]]
        rdata = self.data[self.indptr[i]:self.indptr[i+1]]
        ry = self.d[i]
        
        # Main dot product
        rw = 0.0
        for p in range(rlen):
            rw = rw + rdata[p]*w[rind[p]]
        
        # Loss
        loss = loss + 0.5*(ry - rw)*(ry - rw)
            
    # Normalize and add reg
    loss = loss/self.npoints + 0.5*self.reg*np.dot(w,w)
     
    logger.info("%s loss: %2.6f", txt, loss)  
    return(loss, errors)
