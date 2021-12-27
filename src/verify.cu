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
  const u32 M = 1 << 13;
  CpuTable::Table t_cpu(N * 2, 2);
  CpuTable::UnorderedMap t_stl;

  u32 *key = new u32[N], *qry = new u32[N];
  u32 *res0 = new u32[N], *res1 = new u32[N];

  // auto f = [&]() {
    randomArray(key, N), randomArray(qry, N);

    u32 *sqry = new u32[M];
    randomArray(sqry, M);
    for (u32 i = 0; i < M; i++) qry[sqry[i] % N] = key[sqry[i] % N];
    delete[] sqry;

    t_cpu.clear(), t_stl.clear();
    t_cpu.update(key, N), t_stl.update(key, N);
    t_cpu.query(qry, res0, N), t_stl.query(qry, res1, N);
    bool cmp = std::equal(res0, res0 + N, res1, res1 + N);
    if(!cmp) std::abort();
  // };
  // time_func(f, 10);

  delete[] key, delete[] qry;
  delete[] res0, delete[] res1;
  return;
}
void checkGpuTable() {
  const u32 N = 1 << 20;
  const u32 M = 1 << 13;
  GpuTable::Table t_gpu(N * 2, 2);
  CpuTable::UnorderedMap t_stl;

  u32 *hostKey = new u32[N], *hostQry = new u32[N];
  u32 *deviceKey = coda::malloc<u32>(N), *deviceQry = coda::malloc<u32>(N);
  u32 *hostResult = new u32[N], *hostGpuResult = new u32[N];
  u32 *deviceResult = coda::malloc<u32>(N);

  // auto f = [&]() {
    coda::randomArray(deviceKey, N), coda::randomArray(deviceQry, N), coda::copy(deviceQry, deviceKey, M, coda::D2D);
    coda::copy(hostKey, deviceKey, N, coda::D2H), coda::copy(hostQry, deviceQry, N, coda::D2H);

    t_gpu.clear(), t_stl.clear();
    t_gpu.update(deviceKey, N), t_stl.update(hostKey, N);
    t_gpu.query(deviceQry, deviceResult, N), t_stl.query(hostQry, hostResult, N);
    coda::copy(hostGpuResult, deviceResult, N, coda::D2H);

    bool cmp = std::equal(hostGpuResult, hostGpuResult + N, hostResult, hostResult + N);
    if(!cmp) std::abort();
  // };
  // time_func(f, 10);

  delete[] hostKey, delete[] hostQry, delete[] hostResult, delete[] hostGpuResult;
  coda::free(deviceKey), coda::free(deviceQry), coda::free(deviceResult);
  return;
}

void checkBatchUpdate() {
  // for k>=21, likely to trigger infinity rehash.
  u32 k = 0;
  std::cin >> k;
  const u32 N = 1 << k;
  GpuTable::Table t_gpu(N * 2, 2);

  u32 *deviceKey = coda::malloc<u32>(N);
  u32 *hostKey = new u32[N];
  coda::randomArray(deviceKey, N);
  coda::copy(hostKey, deviceKey, N, coda::D2H);

  // freopen("input","w",stdout);
  // for(u32 i=0;i<N;i++) printf("%u ", hostKey[i]);
  // fflush(stdout);


  CUDA_CALL(cudaDeviceSynchronize());

  Debug() << "GO\n";
  t_gpu.update(deviceKey, N);
  Debug() << "END\n";

  CUDA_CALL(cudaDeviceSynchronize());

  coda::free(deviceKey);
  delete[] hostKey;
  return;
}

int main(){
  checkCpuTable();
  checkGpuTable();
  return 0;
}
