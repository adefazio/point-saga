#cython: profile=False
import logging

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, log, fabs, copysign

import scipy
from scipy.sparse import csr_matrix

cimport loss
from loss cimport Loss

cimport cython

cdef class Loss:
  
  def __init__(self, Xn, double [:] dn, props):
    
    if not isinstance(Xn, csr_matrix):
        raise Exception("Loss requires a CSR matrix")
    
    self.indices = Xn.indices.astype(np.int32)
    self.indptr = Xn.indptr.astype(np.int32)
    self.data = Xn.data.astype(np.float64)
    self.d = dn
    cdef double [:] rdata
    cdef unsigned int i
    
    self.npoints = Xn.shape[0] # points
    self.ndims = Xn.shape[1] # dims
    
    self.reg = props.get("reg", 0.0001)
    
    self.wlist = []
    self.flist = []
    self.errorlist = []
    
    self.avg_wlist = []
    self.avg_flist = []
    self.avg_errorlist = []
    
    logger = logging.getLogger("loss")
    
    # Compute norm squared for each datapoint. Needed for prox.
    self.norm_sq = np.zeros(self.npoints)
    for i in range(self.npoints):
      rdata = self.data[self.indptr[i]:self.indptr[i+1]]
      self.norm_sq[i] = np.dot(rdata, rdata)
      
    norm_sq_mean = np.mean(self.norm_sq)
    logger.info("Squared norm mean: %1.5f. Percentiles:", norm_sq_mean)
    perc = np.percentile(self.norm_sq, [0, 2.5, 25, 50, 75, 97.5, 100])
    logger.info(perc)
    logger.info("Max/mean: %1.2f", perc[6]/norm_sq_mean)
   
  @cython.cdivision(True)
  @cython.boundscheck(False) 
  cpdef double subgradient(self, unsigned int i, double activation):
    raise NotImplementedError( "subgradient" )
    
    
  @cython.cdivision(True)
  @cython.boundscheck(False)
  cpdef tuple prox(self, double gamma, unsigned int i, double activation):
    raise NotImplementedError( "prox" )
      
    
  cpdef compute_loss(self, double[:] w, txt=""):
    raise NotImplementedError( "compute_loss" )
    
  cpdef store_training_loss(self, double [:] w, txt=""):        
    (loss, errors) = self.compute_loss(w, txt)
         
    self.wlist.append(np.copy(w))
    self.flist.append(loss)
    self.errorlist.append(errors) 
