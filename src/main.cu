#include "common.cuh"

#include <functional>
#include <set>
#include <vector>

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
  auto randKeys = mallocDevice<u32>(N);
  auto f = [&]() { batchRandomGen(randKeys, N); };
  auto span = time_func(f, 20u);
  Output() << "takes" << span << "seconds for one generation\n";

  auto *data = new u32[N];
  cudaCopy(data, randKeys, N, CopyKind::D2H);
  std::set<u32> dup;
  while (dup.size() < 100u) dup.insert(rand() % N);

  auto out = Output();
  for (auto x : dup) out << data[x] << "\n";
  return 0;
}
