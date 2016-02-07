from numpy import *
from scipy import sparse
import scipy
import scipy.io
import re
import logging


def readLibSVM(fname, ndata, nfeatures, binary=True, dense=False):
    logger = logging.getLogger("read")
    
    # Read data into a scipy sparse matrix.
    
    n = ndata
    m = nfeatures + 1
    
    if dense:
        X = zeros((n,m))
    else:
        X = sparse.lil_matrix((n, m))
    
    dclass = zeros(n)
    
    h = re.compile('(\d*):(-?\d+\.?\d*)')
    f = open(fname, 'r')
    
    i = 0
    posPoints = 0
    negPoints = 0
    for line in f.xreadlines():
        parts = line.split(' ', 1)
        
        if binary:
            if double(parts[0]) == 1:
                di = 1
            else:
                di = -1
            
            if di == 1:
                posPoints += 1
            else:
                negPoints += 1    
        else:
            di = double(parts[0])
        
         
        dclass[i] = di
        for (j, v) in re.findall(h, parts[1]):
            if int(j) > m:
                raise Exception("Bad column index: %d" % int(j))
            X[i, int(j)] = float(v)
            
        #import pdb; pdb.set_trace()
        
        # libsvm starts at j=1, so we use j=0 as a bias term.
        X[i, 0] = 1.0
        i += 1
        
        if i % 2000 == 0 and i != 0:
            if binary:
                logger.info("--- read %d points, %d positive %d negative", i, posPoints, negPoints)
            else:
                logger.info("--- read %d points", i)
            
      
    logger.info("Finished. %d points.", i) 
    if binary:     
        logger.info("%d positive %d negative", posPoints, negPoints)
    
    f.close()
    
    return (X,dclass)
