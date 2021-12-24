#include "common.cuh"
#include "cuckoo_cpu.cuh"
#include "cuckoo_gpu.cuh"

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

void checkCpuTable() {
  const u32 N = 1 << 20;
  const u32 M = 1 << 18;
  CpuTable::Table t_cpu(N*4, 2);
  CpuTable::UnorderedMap t_stl;

  u32 *key = new u32[N], *qry = new u32[N];
  randomArray(key, N);
  u32 *res0 = new u32[N], *res1 = new u32[N];
  fillZero(res0, N), fillZero(res1, N);

  auto f = [&]() {
    randomArray(qry, N);
    u32 *sqry = new u32[M];
    randomArray(sqry, M);
    for (u32 i = 0; i < M; i++) qry[sqry[i] % N] = key[sqry[i] % N];
    delete[] sqry;

    t_cpu.update(key, N), t_stl.update(key, N);
    t_cpu.query(qry, res0, N), t_stl.query(qry, res0, N);
    bool cmp = std::equal(res0, res0 + M, res1, res1 + M);
    assert(cmp);
  };

  time_func(f, 10);

  delete[] key, delete[] qry;
  delete[] res0, delete[] res1;
  return;
}
i32 main() {
  checkCpuTable();
  return 0;
}
