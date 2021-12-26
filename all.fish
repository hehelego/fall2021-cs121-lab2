#!/usr/bin/fish

set GPU_SM_ARCH 35 # tested on Kepler K80/K40. CUDA 10.0.130
set BUILD_TYPE Release
set BUILD_TYPE Debug

cmake -D CMAKE_CUDA_ARCHITECTURES=$GPU_SM_ARCH -D CMAKE_BUILD_TYPE=$BUILD_TYPE .
make -j8
