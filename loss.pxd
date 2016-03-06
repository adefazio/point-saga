#cython: profile=False

cimport cython
from cpython cimport bool


cdef class Loss:
  cdef int [:] indices, indptr
  cdef double [:] data
  cdef double [:] d
  cdef double [:] norm_sq
  
  cdef int npoints;
  cdef int ndims;
  
  cdef double reg;
  
  cdef public list wlist
  cdef public list flist
  cdef public list errorlist
  
  cdef public list avg_wlist
  cdef public list avg_flist
  cdef public list avg_errorlist
   
  cdef public bool is_differentiable
   
  cpdef double subgradient(self, unsigned int i, double activation)
    
  cpdef double hessianscale(self, unsigned int i, double activation)  
    
  cpdef tuple prox(self, double gamma, unsigned int i, double activation)
    
  cpdef compute_loss(self, double[:] w, txt=?)
    
  cpdef store_training_loss(self, double [:] w, txt=?)
