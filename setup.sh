#!/bin/bash -e

# Use if wget is not installed on your machine
alias wget='curl -O'

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2

bunzip2 rcv1_train.binary.bz2

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2

bunzip2 covtype.libsvm.binary.scale.bz2

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms

# Convert to matlab format matrices using my python code (slow)
ipython datasets/convert_all.py
