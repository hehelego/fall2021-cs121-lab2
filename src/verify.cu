#include "common.cuh"
#include "cuckoo_cpu.cuh"
#include "cuckoo_gpu.cuh"

#include <fstream>
#include <functional>
#include <iomanip>
#include <set>
#include <unordered_map>
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

void checkCpuTable() {
  const u32 N = 1 << 20;
  const u32 M = 1 << 17;
  CpuTable::Table t_cpu(N * 2, 2);
  CpuTable::UnorderedMap t_stl;

  u32 *key = new u32[N], *qry = new u32[N];
  u32 *res0 = new u32[N], *res1 = new u32[N];

  auto f = [&]() {
    randomArrayUnique(key, N), randomArray(qry, N);

    u32 *sqry = new u32[M];
    randomArray(sqry, M);
    for (u32 i = 0; i < M; i++) qry[sqry[i] % N] = key[sqry[i] % N];
    delete[] sqry;

    t_cpu.clear(), t_stl.clear();
    t_cpu.update(key, N), t_stl.update(key, N);
    t_cpu.query(qry, res0, N), t_stl.query(qry, res1, N);
    for (u32 i = 0; i < N; i++) {
      if (bool(res0[i]) != bool(res1[i])) {
        Debug() << "[ERR: CPU cuckoo, STL unordered_map] " << qry[i] << " " << res0[i] << " " << res1[i] << "\n";
        std::abort();
      }
    }
  };
  timeFunc(f, 10);

  delete[] key, delete[] qry;
  delete[] res0, delete[] res1;
  return;
}
void checkGpuTable() {
  const u32 N = 1 << 20;
  const u32 M = 1 << 17;
  GpuTable::Table t_gpu(N * 2, 2);
  CpuTable::UnorderedMap t_stl;

  u32 *hostKey = new u32[N], *hostQry = new u32[N];
  u32 *deviceKey = coda::malloc<u32>(N), *deviceQry = coda::malloc<u32>(N);
  u32 *hostResult = new u32[N], *hostGpuResult = new u32[N];
  u32 *deviceResult = coda::malloc<u32>(N);

  auto f = [&]() {
    coda::randomArrayUnique(deviceKey, N), coda::randomArray(deviceQry, N), coda::copy(deviceQry, deviceKey, M, coda::D2D);
    coda::copy(hostKey, deviceKey, N, coda::D2H), coda::copy(hostQry, deviceQry, N, coda::D2H);

    t_gpu.clear(), t_stl.clear();
    t_gpu.update(deviceKey, N), t_stl.update(hostKey, N);
    t_gpu.query(deviceQry, deviceResult, N), t_stl.query(hostQry, hostResult, N);
    coda::copy(hostGpuResult, deviceResult, N, coda::D2H);

    for (u32 i = 0; i < N; i++) {
      if (bool(hostGpuResult[i]) != bool(hostResult[i])) {
        Debug() << "[ERR: GPU cuckoo, STL unordered_map]" << i << " " << hostQry[i] << " " << hostGpuResult[i] << " "
                << hostResult[i] << "\n";
      }
    }
  };
  timeFunc(f, 10);

  delete[] hostKey, delete[] hostQry, delete[] hostResult, delete[] hostGpuResult;
  coda::free(deviceKey), coda::free(deviceQry), coda::free(deviceResult);
  return;
}

// for K > 20, likely to trigger infinity rehash cycle
// testing result: this is caused by duplicated keys, we need to introduce duplication detection.
void checkBatchUpdate() {
  const u32 K = 24;
  const u32 N = 1 << K;
  GpuTable::Table t_gpu(N * 2, 2);

  u32 *deviceKey = coda::malloc<u32>(N);
  u32 *hostKey = new u32[N];
  coda::randomArrayUnique(deviceKey, N);
  coda::copy(hostKey, deviceKey, N, coda::D2H);

  {
    std::ofstream input("input");
    for (u32 i = 0; i < N; i++) input << hostKey[i] << ' ';
  }
  t_gpu.update(deviceKey, N);
  coda::free(deviceKey);
  delete[] hostKey;
  return;
}

i32 main() {
  Debug() << "check CPU table start\n";
  checkCpuTable();
  Debug() << "check CPU table passed\n";

  Debug() << "check GPU table start\n";
  checkGpuTable();
  Debug() << "check GPU table passed\n";

  Debug() << "check update kernel start\n";
  checkBatchUpdate();
  Debug() << "check update kernel passed\n";
  return 0;
}
