#cython: profile=False
import logging

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, log, fabs, copysign

cimport loss
from loss cimport Loss

cimport cython

@cython.cdivision(True)
cdef inline double prox_newton_step(double y, double x2, double gamma_scaled, double ry):
    cdef double expy = exp(y*ry)
    cdef double sigma = -ry/(1 + expy)
    cdef double gamma_scaled_sigma = gamma_scaled*sigma
    cdef double numerator = gamma_scaled_sigma  + (y-x2)
    cdef double denominator = 1 - ry*gamma_scaled_sigma- gamma_scaled_sigma*sigma

    return(numerator/denominator)


cdef class LogisticLoss(Loss):

  def __init__(self, Xn, double [:] dn, props):
    super(LogisticLoss, self).__init__(Xn, dn, props)
    self.is_differentiable = True
   
  @cython.cdivision(True)
  @cython.boundscheck(False) 
  cpdef double linesearch(self, unsigned int i, double activation, double L):
    cdef double ry = self.d[i]
    cdef unsigned int ls_iters = 0
    
    cdef double Lg = L
    cdef double f_x = self.fval(i, activation)
    cdef double g_i = self.subgradient(i, activation)
    
    cdef double ap = activation - (self.norm_sq[i]*g_i)/Lg
    cdef double fp = self.fval(i, ap)
    
    #print "a=%1.5f, g_i=%1.5f, xns=%1.2f, ap=%1.5f Lg=%1.9f" % (activation, g_i, self.norm_sq[i], ap, Lg)
    #print "fp: %1.8f, fm: %1.8f, f_x=%1.8f" % (fp, f_x - self.norm_sq[i]*g_i*g_i/(2.0*Lg), f_x)
    
    # Finite differences check. Ok.
    #fchange = self.fval(i, activation+1e-8) - self.fval(i, activation)
    #print "fd g: %1.7f" % (fchange/1e-8)
    
    while fp >= f_x - self.norm_sq[i]*g_i*g_i/(2.0*Lg) and ls_iters < 20:
      Lg = 2.0*Lg
      ap = activation - (self.norm_sq[i]*g_i)/Lg
      fp = self.fval(i, ap)
      ls_iters += 1
      #print "! a=%1.5f, g_i=%1.5f, xns=%1.2f, ap=%1.5f, y=%d" % (activation, g_i, self.norm_sq[i], ap, ry)
      #print "! fp: %1.8f, fm: %1.8f, f_x=%1.8f" % (fp, f_x - self.norm_sq[i]*g_i/(2.0*Lg), f_x)
    
    #print " Lg: %1.7f" % Lg
  
    if ls_iters >= 10:
      print " Ls iters very large: %d, Lg=%1.7f" % (ls_iters, Lg)
      
        
    return Lg
   
  @cython.cdivision(True)
  @cython.boundscheck(False) 
  cpdef double fval(self, unsigned int i, double activation):
    cdef double ry = self.d[i]
   
    return(log(1+exp(-activation*ry)))
   
  @cython.cdivision(True)
  @cython.boundscheck(False) 
  cpdef double subgradient(self, unsigned int i, double activation):
    cdef double ry = self.d[i]
     
    return(-ry/(1 + exp(activation*ry)))
    
  @cython.cdivision(True)
  @cython.boundscheck(False)
  cpdef tuple prox(self, double gamma, unsigned int i, double activation):
    cdef double sg, xp_old
    cdef double gamma_scaled = gamma*self.norm_sq[i]
    cdef double ry = self.d[i]
    #logger = logging.getLogger("logisticloss")
  
    cdef double xp = 0.0
  
    # 3 might be enough in practice, more for badly conditioned problems
    for p in range(12): 
      xp_old = xp
      xp = xp - prox_newton_step(xp, activation, gamma_scaled, ry)
      #logger.info("p: %d, xp_old: %1.16f, xp_new: %1.16f, a: %1.8f", p, xp_old, xp, activation)
  
    sg = (activation - xp)/gamma_scaled
    
    return((xp, sg))
      
    
  cpdef compute_loss(self, double[:] w, txt=""):
    
    logger = logging.getLogger("logisticloss")
    
    cdef double [:] rdata
    cdef int [:] rind
    cdef double rw, ry, pointloss
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
        
        # Numerically stable loss computation
        if ry*rw > 0:
          pointloss = log(1 + exp(-ry*rw))
        else:
          pointloss = -ry*rw + log(1 + exp(ry*rw))

        loss = loss + pointloss

        if ry*rw <= 0:
            errors = errors + 1
            
    # Normalize and add reg
    loss = loss/self.npoints + 0.5*self.reg*np.dot(w,w)
     
    logger.info("%s loss: %2.12f errors: %d (%2.3f percent)" % (
         txt, loss, errors,  100.0*errors/(0.0+self.npoints)))  
    return(loss, errors)
