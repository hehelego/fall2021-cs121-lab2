#include "common.cuh"
#include "cuckoo_gpu.cuh"

#include <functional>
#include <iomanip>
#include <set>
#include <vector>

struct TimeResult {
  double dt_ms; // elapsed time in microseconds (1e-6 second)
  bool valid;   // too much rehash, does not terminate
  TimeResult(double t, bool v) : dt_ms(t), valid(v) {}
  static TimeResult valid_time(double t) { return TimeResult(t, true); }
  static TimeResult does_not_terminate() { return TimeResult(0, false); }
};
inline TimeResult operator+(const TimeResult &tr1, const TimeResult &tr2) {
  return TimeResult(tr1.dt_ms + tr2.dt_ms, tr1.valid && tr2.valid);
}
template <typename T> inline TimeResult operator*(const TimeResult &tr1, const T k) {
  return TimeResult(tr1.dt_ms * k, tr1.valid);
}
template <typename T> inline TimeResult operator*(const T k, const TimeResult &tr1) {
  return TimeResult(tr1.dt_ms * k, tr1.valid);
}
template <typename T> inline TimeResult operator/(const TimeResult &tr1, const T k) {
  return TimeResult(tr1.dt_ms / k, tr1.valid);
}
inline std::ostream &operator<<(std::ostream &os, const TimeResult &tr) {
  if (tr.valid) {
    os << "Time(" << tr.dt_ms << ")";
  } else {
    os << "InvalidTime";
  }
  return os;
}

TimeResult timeOnce(std::function<void()> func) {
  static Timer timer;
  GpuTable::REHASH_COUNT = 0;
  CUDA_CALL(cudaDeviceSynchronize()), timer.start();
  func();
  CUDA_CALL(cudaDeviceSynchronize()), timer.end();
  if (GpuTable::TOO_MUCH_REHASH) return TimeResult::does_not_terminate();
  TimeResult::valid_time(timer.delta_ms());
}
TimeResult timeFunc(std::function<void()> pre, std::function<void()> func, std::function<void()> post, u32 runs = 5) {
  TimeResult total_cost = TimeResult::valid_time(0);
  for (u32 i = 0; i < runs; i++) {
    pre();
    total_cost = total_cost + timeOnce(func);
    post();
  }
  return total_cost / runs;
}

void benchmark_task1() {
  GpuTable::Table gpu_table(1 << 25, 2);

  u32 *h_keys = new u32[1 << 24], *d_keys = coda::malloc<u32>(1 << 24);
  coda::randomArrayUnique(d_keys, 1 << 24), coda::copy(h_keys, d_keys, 1 << 24, coda::D2H);

  Output() << "benchmark 1: insertion\n";
  for (u32 i = 10; i < 24; i++) {
    auto gpu_time = timeFunc([&]() { gpu_table.clear(); }, [&]() { gpu_table.update(d_keys, 1 << i); }, []() {});

    Output() << std::fixed << std::setprecision(6) << i << "\t\t" << gpu_time << "\n";
  }

  delete[] h_keys, coda::free(d_keys);
}
void benchmark_task2() {
  GpuTable::Table gpu_table(1 << 25, 2);

  u32 *h_keys = new u32[1 << 24], *d_keys = coda::malloc<u32>(1 << 24);
  coda::randomArrayUnique(d_keys, 1 << 24), coda::copy(h_keys, d_keys, 1 << 24, coda::D2H);
  gpu_table.update(d_keys, 1 << 24);
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

    auto gpu_time = timeFunc([]() {}, [&]() { gpu_table.query(d_qrys, d_res, 1 << 24); }, []() {});

    Output() << std::fixed << std::setprecision(6) << i << "\t\t" << gpu_time << "\n";
  }

  delete[] h_qrys, delete[] h_res, coda::free(d_qrys), coda::free(d_res);
}
void benchmark_task3() {
  u32 *h_keys = new u32[1 << 24], *d_keys = coda::malloc<u32>(1 << 24);
  coda::randomArrayUnique(d_keys, 1 << 24), coda::copy(h_keys, d_keys, 1 << 24, coda::D2H);

  Output() << "benchmark 3: insertion with high load factor\n";
  for (u32 i = 1; i <= 10; i++) {
    u32 cnt = (1 << 24) * i / 10;
    auto gpu_time = timeFunc([]() {},
                             [&]() {
                               GpuTable::Table gpu_table((1 << 24) + cnt, 2);
                               gpu_table.update(d_keys, 1 << 24);
                             },
                             []() {});
    Output() << std::fixed << std::setprecision(6) << i << "\t\t" << gpu_time << "\n";
  }

  // Out Of Memory or do not terminate
  {
    u32 cnt = (1 << 24) * 1 / 100;
    auto gpu_time = timeFunc([]() {},
                             [&]() {
                               GpuTable::Table gpu_table((1 << 24) + cnt, 2);
                               gpu_table.update(d_keys, 1 << 24);
                             },
                             []() {});
    Output() << std::fixed << std::setprecision(6) << "1.01"
             << "\t\t" << gpu_time << "\n";
  }

  // Out Of Memory or do not terminate
  {
    u32 cnt = (1 << 24) * 2 / 100;
    auto gpu_time = timeFunc([]() {},
                             [&]() {
                               GpuTable::Table gpu_table((1 << 24) + cnt, 2);
                               gpu_table.update(d_keys, 1 << 24);
                             },
                             []() {});
    Output() << std::fixed << std::setprecision(6) << "1.02"
             << "\t\t" << gpu_time << "\n";
  }

  // Out Of Memory or do not terminate
  {
    u32 cnt = (1 << 24) * 5 / 100;
    auto gpu_time = timeFunc([]() {},
                             [&]() {
                               GpuTable::Table gpu_table((1 << 24) + cnt, 2);
                               gpu_table.update(d_keys, 1 << 24);
                             },
                             []() {});
    Output() << std::fixed << std::setprecision(6) << "1.05"
             << "\t\t" << gpu_time << "\n";
  }
  delete[] h_keys, coda::free(d_keys);
}

void benchmark_task4() {
  u32 *h_keys = new u32[1 << 24], *d_keys = coda::malloc<u32>(1 << 24);
  coda::randomArrayUnique(d_keys, 1 << 24), coda::copy(h_keys, d_keys, 1 << 24, coda::D2H);

  Output() << "benchmark 4: eviction chain limit\n";
  for (double i = 1; i < 8; i += 0.5) {
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
