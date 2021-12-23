#!/usr/bin/fish

set GPU_SM_ARCH 37 # tested on Kepler K80

cmake -D CMAKE_CUDA_ARCHITECTURES=35 -D CMAKE_BUILD_TYPE=Release .
make -j8
