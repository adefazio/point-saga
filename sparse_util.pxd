#cython: profile=False

cdef void lagged_update(long k, double[:] x, double[:] g, unsigned int[:] lag, 
                          int[:] yindices, int ylen, double[:] lag_scaling, double a)
   
cdef void lagged_update_with_xbar(long k, double[:] x, double[:] xbar, double alpha_k, 
                          double[:] g, unsigned int[:] lag, 
                          int[:] yindices, int ylen, double[:] lag_scaling, double a)                       
                          
cdef void add_weighted(double[:] x, double[:] ydata , int[:] yindices, int ylen, double a)

cdef double spdot(double[:] x, double[:] ydata , int[:] yindices, int ylen)
                          
cdef void add_weighted_diag(double[:] x, double[:] hk, double[:] ydata , int[:] yindices, int ylen, double a)

cdef void lagged_update_diag(long k, double[:] x, double[:] hk, double[:] g, unsigned int[:] lag, 
                          int[:] yindices, int ylen, double[:] lag_scaling, double a)
