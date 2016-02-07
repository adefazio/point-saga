#!/bin/bash

rm optimization/*.c
rm optimization/*.so
rm *.c
rm *.so
python setup.py clean
python setup.py build_ext --inplace
