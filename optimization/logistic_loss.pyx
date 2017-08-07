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
    # The overkill here is so I don't have to think about it.
    for p in range(12):
      xp_old = xp
      xp = xp - prox_newton_step(xp, activation, gamma_scaled, ry)
      #logger.info("p: %d, xp_old: %1.16f, xp_new: %1.16f, a: %1.8f", p, xp_old, xp, activation)

    sg = (activation - xp)/gamma_scaled

    return((xp, sg))

  @cython.cdivision(True)
  @cython.boundscheck(False)
  cpdef double hessian(self, unsigned int i, double activation):
    cdef double ry = self.d[i]
    cdef expna = exp(-activation*ry)
    cdef sigma = 1.0/(1.0 + expna)
    return sigma*(1-sigma)

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
