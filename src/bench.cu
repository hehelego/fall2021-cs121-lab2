#include "common.cuh"
#include "cuckoo_cpu.cuh"
#include "cuckoo_gpu.cuh"

#include <functional>
#include <iomanip>
#include <set>
#include <vector>

double timeOnce(std::function<void()> func) {
  static Timer timer;
  CUDA_CALL(cudaDeviceSynchronize());
  timer.start();
  func();
  CUDA_CALL(cudaDeviceSynchronize());
  timer.end();
  return timer.deltaInSeconds();
}
double timeFunc(std::function<void()> func, u32 runs) {
  double s = 0;
  for (u32 i = 0; i < runs; i++) s += timeOnce(func);
  return s / runs;
}

i32 main() { return 0; }