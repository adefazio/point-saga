#cython: profile=False

cdef void lagged_update(long k, double[:] x, double[:] g, unsigned int[:] lag, 
                          int[:] yindices, int ylen, double[:] lag_scaling, double a)
   
cdef void lagged_update_with_xbar(long k, double[:] x, double[:] xbar, double alpha_k, 
                          double[:] g, unsigned int[:] lag, 
                          int[:] yindices, int ylen, double[:] lag_scaling, double a)                       
                          
cdef void add_weighted(double[:] x, double[:] ydata , int[:] yindices, int ylen, double a)

cdef double spdot(double[:] x, double[:] ydata , int[:] yindices, int ylen)

cdef double flat_unlag(long m, long k, double[:] xk, double[:] gk, unsigned int[:] lag, double a)

cdef void flat_lagged_update(long k, double[:] x, double[:] g, unsigned int[:] lag, 
                          int[:] yindices, int ylen, double a)

cdef void flat_single_step(long k, double[:] x, double[:] g, unsigned int[:] lag, 
            int[:] yindices, int ylen, double a)

ctypedef struct interlaced:
  double x
  double g
  long lag
