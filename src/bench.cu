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
double timeFunc(std::function<void()> pre, std::function<void()> func, std::function<void()> post, u32 runs = 5) {
  double s = 0;
  for (u32 i = 0; i < runs; i++) {
    pre();
    s += timeOnce(func);
    post();
  }
  return s / runs;
}

void benchmark_task1() {
  GpuTable::Table gpu_table(1 << 25, 2);
  CpuTable::UnorderedMap cpu_table;

  u32 *h_keys = new u32[1 << 24], *d_keys = coda::malloc<u32>(1 << 24);
  randomArrayUnique(h_keys, 1 << 24), coda::copy(d_keys, h_keys, 1 << 24, coda::H2D);

  Output() << "benchmark 1: insertion\n";
  for (u32 i = 10; i < 24; i++) {
    auto cpu_time = timeFunc([&]() { cpu_table.clear(); }, [&]() { cpu_table.update(h_keys, 1 << i); }, []() {});
    auto gpu_time = timeFunc([&]() { gpu_table.clear(); }, [&]() { gpu_table.update(d_keys, 1 << i); }, []() {});

    Output() << std::fixed << std::setprecision(6) << i << "\t\t" << cpu_time << "\t\t" << gpu_time << "\n";
  }

  delete[] h_keys, coda::free(d_keys);
}
void benchmark_task2() {
  GpuTable::Table gpu_table(1 << 25, 2);
  CpuTable::UnorderedMap cpu_table;

  u32 *h_keys = new u32[1 << 24], *d_keys = coda::malloc<u32>(1 << 24);
  randomArrayUnique(h_keys, 1 << 24), coda::copy(d_keys, h_keys, 1 << 24, coda::H2D);
  cpu_table.update(h_keys, 1 << 24), gpu_table.update(d_keys, 1 << 24);
  delete[] h_keys, coda::free(d_keys);

  std::mt19937 rng(randomSeed());
  u32 *h_qrys = new u32[1 << 24], *d_qrys = coda::malloc<u32>(1 << 24);
  u32 *h_res = new u32[1 << 24], *d_res = coda::malloc<u32>(1 << 24);
  coda::copy(h_qrys, h_keys, 1 << 24, coda::H2H);

  Output() << "benchmark 2: query\n";
  for (u32 i = 0; i <= 10; i++) {
    u32 cnt = (1 << 24) * i / 10;
    for (u32 j = 0; j < cnt; j++) h_qrys[j] = h_keys[rng() % (1 << 24)];
    coda::copy(d_qrys, h_qrys, 1 << 24, coda::H2D);

    auto cpu_time = timeFunc([]() {}, [&]() { cpu_table.query(h_qrys, h_res, 1 << 24); }, []() {});
    auto gpu_time = timeFunc([]() {}, [&]() { gpu_table.query(d_qrys, d_res, 1 << 24); }, []() {});

    Output() << std::fixed << std::setprecision(6) << i << "\t\t" << cpu_time << "\t\t" << gpu_time << "\n";
  }

  delete[] h_qrys, delete[] h_res, coda::free(d_qrys), coda::free(d_res);
}
void benchmark_task3() {

  u32 *h_keys = new u32[1 << 24], *d_keys = coda::malloc<u32>(1 << 24);
  randomArrayUnique(h_keys, 1 << 24), coda::copy(d_keys, h_keys, 1 << 24, coda::H2D);

  Output() << "benchmark 3: insertion with high load factor\n";
  for (u32 i = 1; i <= 10; i++) {
    u32 cnt = (1 << 24) * i / 10;
    auto cpu_time = timeFunc([]() {},
                             [&]() {
                               CpuTable::UnorderedMap cpu_table;
                               cpu_table.update(h_keys, 1 << 24);
                             },
                             []() {});
    auto gpu_time = timeFunc([]() {},
                             [&]() {
                               GpuTable::Table gpu_table((1 << 24) + cnt, 2);
                               gpu_table.update(d_keys, 1 << 24);
                             },
                             []() {});
    Output() << std::fixed << std::setprecision(6) << i << "\t\t" << cpu_time << "\t\t" << gpu_time << "\n";
  }

  // Out Of Memory or do not terminate
  // {
  //   u32 cnt = (1 << 24) * 1 / 100;
  //   auto gpu_time = timeFunc([]() {},
  //                            [&]() {
  //                              GpuTable::Table gpu_table((1 << 24) + cnt, 2);
  //                              gpu_table.update(d_keys, 1 << 24);
  //                            },
  //                            []() {});
  //   Output() << std::fixed << std::setprecision(6) << "1.01" << "\t\t" << gpu_time << "\n";
  // }

  // Out Of Memory or do not terminate
  // {
  //   u32 cnt = (1 << 24) * 2 / 100;
  //   auto gpu_time = timeFunc([]() {},
  //                            [&]() {
  //                              GpuTable::Table gpu_table((1 << 24) + cnt, 2);
  //                              gpu_table.update(d_keys, 1 << 24);
  //                            },
  //                            []() {});
  //   Output() << std::fixed << std::setprecision(6) << "1.01" << "\t\t" << gpu_time << "\n";
  // }

  // Out Of Memory or do not terminate
  // {
  //   u32 cnt = (1 << 24) * 5 / 100;
  //   auto gpu_time = timeFunc([]() {},
  //                            [&]() {
  //                              GpuTable::Table gpu_table((1 << 24) + cnt, 2);
  //                              gpu_table.update(d_keys, 1 << 24);
  //                            },
  //                            []() {});
  //   Output() << std::fixed << std::setprecision(6) << "1.01" << "\t\t" << gpu_time << "\n";
  // }
  delete[] h_keys, coda::free(d_keys);
}

void benchmark_task4() {

  u32 *h_keys = new u32[1 << 24], *d_keys = coda::malloc<u32>(1 << 24);
  randomArrayUnique(h_keys, 1 << 24), coda::copy(d_keys, h_keys, 1 << 24, coda::H2D);

  Output() << "benchmark 4: eviction chain limit\n";
  for (double i = 1; i < 8; i += 0.2) {
    auto gpu_time = timeFunc([]() {},
                             [&]() {
                               GpuTable::Table gpu_table(1 << 25, 2, i);
                               gpu_table.update(d_keys, 1 << 24);
                             },
                             []() {});
    Output() << std::fixed << std::setprecision(6) << i << "\t\t" << gpu_time << "\n";
  }

  delete[] h_keys, coda::free(d_keys);
}
i32 main() {
  benchmark_task1();
  benchmark_task2();
  benchmark_task3();
  benchmark_task4();
  return 0;
}