#cython: profile=False
import logging

import numpy as np
cimport numpy as np

cimport loss
from loss cimport Loss

cimport cython

cdef class HingeLoss(Loss):

  def __init__(self, Xn, double [:] dn, props):
    super(HingeLoss, self).__init__(Xn, dn, props)
    self.is_differentiable = False
   
  @cython.cdivision(True)
  @cython.boundscheck(False)  
  cpdef double subgradient(self, unsigned int i, double activation):
    cdef double ry = self.d[i]
     
    if ry*activation < 1:
      return(-ry)
    else:
      return(0)

  @cython.cdivision(True)
  @cython.boundscheck(False)   
  cpdef tuple prox(self, double gamma, unsigned int i, double activation):
    cdef double ry = self.d[i]
    
    # Prox operation
    cdef double s = (1.0 - ry*activation)/(gamma*self.norm_sq[i])
    
    if s >= 1:
        cnew = -ry
    elif s <= 0:
        cnew = 0
    else:
        cnew = -ry*s
    
    return((activation-gamma*cnew, cnew))
    
  cpdef compute_loss(self, double[:] w, txt=""):
    
    logger = logging.getLogger("hingeloss")
    
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
        
        #logger.info("ry: %2.2e, rw: %2.2e", ry, rw)
        
        if ry*rw < 1:
            loss = loss + 1-ry*rw 
        if ry*rw <= 0:
            errors = errors + 1
            
    # Normalize and add reg
    loss = loss/self.npoints + 0.5*self.reg*np.dot(w,w)
     
    logger.info("%s loss: %2.6f errors: %d (%2.3f percent)" % (
         txt, loss, errors,  100.0*errors/(0.0+self.npoints)))  
    return(loss, errors)
