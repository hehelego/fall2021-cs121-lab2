#include "common.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <functional>

double time_once(std::function<void()> func) {
  static Timer timer;
  CUDA_CALL(cudaDeviceSynchronize());
  timer.start();
  func();
  CUDA_CALL(cudaDeviceSynchronize());
  timer.end();
  return timer.time_in_second();
}

i32 main() { return 0; }
