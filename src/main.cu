#include "common.cuh"

#include <functional>
#include <iomanip>
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
const u32 SAMPLE = 30;
const u32 SEED = 19260917;

__global__ void batchHash(u32 *d_k, u32 *d_h, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n) d_h[i] = xxHash32(SEED, d_k[i]);
}
i32 test_rand_hash() {
  auto randKeys = coda::malloc<u32>(N), hashKeys = coda::malloc<u32>(N);
  coda::fillZero(randKeys, N), coda::fillZero(hashKeys, N);
  batchRandomGen(randKeys, N);

  auto f = [&]() {
    batchHash<<<N / 1024, 1024>>>(randKeys, hashKeys, N);
    Debug() << "once\n";
  };
  auto span = time_func(f, 10u);
  Output() << "takes " << std::fixed << std::setprecision(6) << span << " seconds for one run\n";

  auto keys = new u32[N], hkeys = new u32[N];
  coda::copy(keys, randKeys, N, coda::CopyKind::D2H);
  coda::copy(hkeys, hashKeys, N, coda::CopyKind::D2H);

  std::set<u32> dup;
  while (dup.size() < SAMPLE) dup.insert(rand() % N);

  auto out = Output();
  for (auto i : dup) {
    out << "key:" << keys[i] << "\t";
    out << "hash(gpu):" << hkeys[i] << "\t";
    out << "hash(cpu):" << xxHash32(SEED, keys[i]) << "\n";
  };

  coda::free(randKeys), coda::free(hashKeys);
  delete[] keys, delete[] hkeys;
  return 0;
}

i32 main() {
  test_rand_hash();
  return 0;
}
