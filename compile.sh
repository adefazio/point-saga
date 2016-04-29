#!/bin/bash

python setup.py build_ext --inplace
dsymutil sagafast.so -o sagafast.so.dSYM
