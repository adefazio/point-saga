
from datasets.libsvm2mat import libsvm2mat

libsvm2mat(
    fname="rcv1_train.binary", 
    ndata=20242,
    nfeatures=47236)

libsvm2mat(
    fname="australian_scale", 
    ndata=690,
    nfeatures=14)

#libsvm2mat(
#    fname="australian",
#    ndata=690,
#    nfeatures=14)

libsvm2mat(
    fname="covtype.libsvm.binary.scale",
    ndata=581012,
    nfeatures=54)

libsvm2mat(
    fname="mushrooms",
    ndata=8124,
    nfeatures=112)
