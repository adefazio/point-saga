
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdlib cimport rand
from libc.math cimport floor

from cython cimport view

from libcpp.vector cimport vector

ctypedef struct SampleInfo:
    unsigned int idx
    unsigned int level
    unsigned int idx_in_level
    double weight

cdef class FastSampler:

    cdef unsigned int nentries
    cdef unsigned int max_entries

    cdef double max_value
    cdef double min_value

    cdef int top_level
    cdef int bottom_level
    cdef unsigned int nlevels

    cdef double total_weight
    cdef double [::1] weights
    cdef double [::1] level_weights
    cdef vector[vector[int]] level_buckets
    cdef double [::1] level_max
    
    #########
    cdef add(self, unsigned int idx, double weight)
    cdef SampleInfo _sample(self)
    cdef unsigned int sample(self)
    cdef unsigned int sampleAndRemove(self)
    cdef getWeight(self, unsigned int idx)
