#!/bin/bash

THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True' python -u $1 2>&1 | tee $(basename $1 .py).log
