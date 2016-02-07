import numpy as np
import scipy.io
from datasets.read_libsvm import *
import logging
import logging.config
import os

def libsvm2mat(fname, ndata, nfeatures, binary=True):

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    logger = logging.getLogger("opt")

    logger.info("Starting data read")

    (X, d) = readLibSVM(fname=fname, ndata=ndata, nfeatures=nfeatures, binary=binary)

    # Permute rows
    #logger.info("Permuting rows")
    #np.random.seed(42)
    #perm = np.random.permutation(ndata)
    #X = X[perm, :]
    #d = d[perm]

    logger.info("Convert dataset to CSC, in transposed form")
    X = X.tocsr().transpose()

    logger.info("Saving full dataset ...")

    # Save in matlab format
    scipy.io.savemat(
        file_name="%s.mat" % fname,
        mdict={'X': X, 'd': d},
        do_compression=True,
        format='5',
        oned_as='row')
    
    logger.info("Done!!!")
