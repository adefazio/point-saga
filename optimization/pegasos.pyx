import logging
import scipy
from scipy.sparse import csr_matrix
import random

from get_loss import getLoss

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, log, fabs
from cpython cimport bool

STUFF = "Hi"

cimport cython
from cython.view cimport array as cvarray

ctypedef np.float32_t dtype_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def pegasos(Xn, double [:] dn, props):
            
    cdef int [:] indices, indptr
    cdef double [:] data
    cdef int i
    
    if not isinstance(Xn, csr_matrix):
        raise Exception("Pegasos requires a CSR matrix")
        
    indices = Xn.indices.astype(np.int32)
    indptr = Xn.indptr.astype(np.int32)
    data = Xn.data.astype(np.float64)
                
    #cdef double [:,::1] X = Xn
    cdef double [:] d = dn
    
    cdef double [:] rdata
    cdef int [:] rind
    cdef double t, eta, activation, update_scale, mu, sg
    cdef int p,r,j, rlen
    cdef int processed = 0
    cdef int iteration = 0
    
    losses = []
                
    logger = logging.getLogger("pegasos")

    cdef int npoints = Xn.shape[0] # points
    cdef int ndims = Xn.shape[1] # dims

    cdef double stepSize = props.get("stepSize", 0.9)
    cdef int maxIters = props.get("passes", 10)
    cdef int averageAfter = props.get("averageAfter", 1) # in points not epochs
    cdef double reg = props.get("reg", 0.0001)
    cdef bool improved_averaging =  props.get("improvedAveraging", True)
    cdef bool return_averaged = props.get("returnAveraged", False)

    loss = getLoss(Xn, d, props)

    cdef double [:] w = np.zeros(ndims)
    cdef double [:] wbar = np.zeros(ndims)
    cdef double [:] wbar_test = np.zeros(ndims)
    cdef double [:] wbar_interm = np.zeros(ndims)
    cdef double wscale = 1.0
    
    cdef double alpha_k = 0
    cdef double beta_k = 1
    # everywhere the true w should be we instead put w*wscale.

    logger.info("Pegasos starting, npoints=%d, ndims=%d, stepSize: %2.3e", npoints, ndims, stepSize)
    if return_averaged:
      logger.info("Averaging on, skipping first %d, and improved_averaging: %r", averageAfter, improved_averaging)
    
    loss.store_training_loss(w)   
    
    while iteration < maxIters:
        for j in range(npoints):
            i = np.random.randint(0, npoints)
            
            t = j+iteration*npoints+1
            eta = stepSize/t # Step size
            
            rlen = indptr[i+1]-indptr[i]
            rind = indices[indptr[i]:indptr[i+1]]
            rdata = data[indptr[i]:indptr[i+1]]
            ry = d[i]
            
            if processed >= averageAfter:
              if improved_averaging:
                mu = 2.0/(t+2)
              else:
                mu = 1.0/(t+1) # Averaging constant.
            else:
              mu = 1.0 
              alpha_k = 0
              beta_k = 1
            
            # Main dot product
            activation = 0.0
            for p in range(rlen):
                activation = activation + rdata[p]*w[rind[p]]
                
            activation = activation*wscale
                
            wscale = wscale*(1-eta*reg)
            
            #logger.info("ry: %2.2e, activation: %2.2e, eta: %2.2e, reg: %2.2e, t: %2.2e", ry, activation, eta, reg, t)
            
            sg = loss.subgradient(i, activation)
            
            if sg != 0.0:
              update_scale = eta*sg/wscale
              
              for p in range(rlen):
                  w[rind[p]] = w[rind[p]] - update_scale*rdata[p]
                  wbar_interm[rind[p]] = wbar_interm[rind[p]] + alpha_k*update_scale*rdata[p]
            
            #if ry*activation < 1:
                # wtrue = w*wscale
                # w*wscale <- w*wscale*(1-eta*reg) + chg
                # w*wscale <- w*wscale*(1-eta*reg); w*wscale_up += chg
                # therefore w+= chg/wscale_up 
            #    update_scale = eta*ry/wscale
                
            #    for p in range(rlen):
            #        w[rind[p]] = w[rind[p]] + update_scale*rdata[p]
            #        wbar_interm[rind[p]] = wbar_interm[rind[p]] - alpha_k*update_scale*rdata[p]
            #if ry*activation >= 1:
            #
            ###
            
            if False:
              for p in range(ndims):
                wbar_test[p] = wbar_test[p] + mu*(wscale*w[p] - wbar_test[p])
            
            #logger.info("Alpha: %1.1e, beta: %1.1e, processed: %1.1e, mu: %1.1em wscale: %1.10e", alpha_k, beta_k, processed, mu, wscale)
        
            # Handle special case
            if mu >= 1:
              alpha_k = wscale
              beta_k = 1
            else:
              beta_k = beta_k/(1.0-mu)
              alpha_k = alpha_k + mu*beta_k*wscale
            processed = processed + 1
        
        iteration = iteration + 1
        logger.info("Epoch %d finished" % iteration)
        
        #logger.info("Alpha: %1.1e, beta: %1.1e, processed: %1.1e, mu: %1.1em wscale: %1.1e", alpha_k, beta_k, processed, mu, wscale)
        for p in range(ndims):
            wbar[p] = (wbar_interm[p] + alpha_k*w[p])/beta_k
            wbar_interm[p] = wbar[p]
            
        alpha_k = 0 # after wscale changed.
        beta_k = 1
        
        # renormalize w
        for p in range(ndims):
            w[p] = w[p]*wscale
        wscale = 1.0
        
        # Compute end of epoch error rate and function value
        if return_averaged:
          loss.store_training_loss(wbar, "wbar:")
        else:
          loss.store_training_loss(w, "w:")
        #loss.compute_loss(wbar_test, "test:")  
        #loss.compute_loss(wbar, "wbar:")  
        
    logger.info("Pegasos finished")

    return {'wlist': loss.wlist, 'flist': loss.flist, 'errorlist': loss.errorlist}
