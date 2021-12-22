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
double time_func(std::function<void()> func, u32 runs) {
  double s = 0;
  for (u32 i = 0; i < runs; i++) s += time_once(func);
  return s / runs;
}

const u32 N = 1 << 25;
i32 main() {
  auto rand_keys = mallocDevice<u32>(N);
  auto f = [&]() { batchRandomGen(rand_keys, N); };
  auto span = time_func(f, 20u);
  Output() << "takes" << span << "seconds for one generation\n";
  return 0;
}
