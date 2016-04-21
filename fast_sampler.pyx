
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdlib cimport rand
from libc.math cimport floor

from cython cimport view

from libcpp.vector cimport vector

cdef extern from "random_fast.cpp" nogil:
    cdef double uniform_fast()

cdef extern from "<cmath>" nogil:
    cdef double frexp(double x, int* exp)

cdef class FastSampler:

    def __init__(self, unsigned int max_entries, double max_value=100, double min_value=1):
        cdef int i
        print "Setting up sampler ..."
        
        #cdef double exponent
        self.nentries = 0
        self.max_entries = max_entries
        self.max_value = max_value
        self.min_value = min_value
        
        frexp(max_value, &self.top_level)
        frexp(min_value, &self.bottom_level)
        
        self.nlevels = 1 + self.top_level - self.bottom_level
        print "Creating with %d levels" % self.nlevels

        self.total_weight = 0.0
        self.weights = np.zeros(max_entries, dtype='d')


        self.level_weights = np.zeros(self.nlevels, dtype='d')
        self.level_max = np.zeros(self.nlevels, dtype='d')
        for i in range(self.nlevels):
            #print "Foo"
            #print "level i=%d, levelmax: %1.7f" % (i, pow(2, self.top_level-i))
            self.level_buckets.push_back(vector[int]())
            #print "bar"
            self.level_max[i] = pow(2.0, self.top_level-i)
            
        print "Sampler created with %d to levels" % self.nlevels
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef add(self, unsigned int idx, double weight):
        cdef int raw_level
        cdef unsigned int level
        
        #if weight > self.max_value:
        #  weight = self.max_value
        if weight < self.min_value:
          weight = self.min_value + 1e-16
        if weight > self.max_value or weight < self.min_value:
            raise Exception("Weight out of range: %1.2e" % weight)
        
        if idx >= self.max_entries:
            raise Exception("Bad index: %s", idx)
        
        self.nentries += 1
        self.total_weight += weight
        
        self.weights[idx] = weight
        
        frexp(weight, &raw_level)
        level = self.top_level - raw_level
        
        self.level_weights[level] += weight
        self.level_buckets[level].push_back(idx)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef SampleInfo _sample(self):
        cdef double u, u_lvl, level_max, idx_weight, inner_sample, inner_sample_floor
        cdef unsigned int idx, level, level_size, idx_in_level
        cdef double cumulative_weight = 0
        cdef bool reject
        
        u = uniform_fast()*self.total_weight
        
        # Sample a level using the CDF method
        for level in range(self.nlevels):
            cumulative_weight += self.level_weights[level]
            if u < cumulative_weight:
                break
        
        # Now sample within the level using rejection sampling
        level_size = self.level_buckets[level].size()
        level_max = self.level_max[level]
        reject = True
        while reject:
            inner_sample = uniform_fast()*level_size
            inner_sample_floor = floor(inner_sample)
            
            idx_in_level = <int>inner_sample_floor
            idx = self.level_buckets[level][idx_in_level]
            idx_weight = self.weights[idx]
            
            u_lvl = level_max*(inner_sample - inner_sample_floor)
            if u_lvl <= idx_weight:
                reject = False
        
        return SampleInfo(idx, level, idx_in_level, idx_weight)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef unsigned int sample(self):
        return self._sample().idx
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef unsigned int sampleAndRemove(self):
        cdef unsigned int swap_idx
        
        cdef SampleInfo s = self._sample()
        
        # Remove it
        self.total_weight -= s.weight
        self.level_weights[s.level] -= s.weight
        # Swap with last element for efficent delete
        swap_idx = self.level_buckets[s.level].back()
        self.level_buckets[s.level].pop_back()
        self.level_buckets[s.level][s.idx_in_level] = swap_idx
        self.nentries -= 1
        
        return s.idx

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef getWeight(self, unsigned int idx):
        return self.weights[idx]
