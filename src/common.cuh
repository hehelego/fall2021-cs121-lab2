#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// For printing debug log
struct Debug {
  std::ostream &stream;
  Debug(std::ostream &stream = std::cerr) : stream(stream) {}
  template <typename T> inline Debug &operator<<(const T &t) {
#if (DEBUG == 0)
    stream << t;
#else
    (void)t;
#endif
    return *this;
  }
};
// For reporting performance
struct Output {
  std::ostream &stream;
  Output(std::ostream &stream = std::cout) : stream(stream) {}
  template <typename T> inline Output &operator<<(const T &t) {
    stream << t;
    return *this;
  }
};

using u8 = std::uint8_t;
using i8 = std::int8_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;

// For timing
struct Timer {
  std::chrono::high_resolution_clock::time_point _start, _end;
  Timer() {}
  inline void start() { _start = std::chrono::high_resolution_clock::now(); }
  inline void end() { _end = std::chrono::high_resolution_clock::now(); }
  inline double time_in_second() const {
    auto interval = _end - _start;
    return std::chrono::duration_cast<std::chrono::microseconds>(interval).count() / 1000.0;
  }
};

template <typename T>
//  \[ \lfloor \frac{a}{b} \rfloor \]
inline T div_floor(T a, T b) {
  return (a / b);
}
template <typename T>
//  \[ \lceil \frac{a}{b} \rceil \]
inline T div_ceil(T a, T b) {
  return (a + b - 1) / b;
}

#define CUDA_CALL(cuda_call)                                                                                                     \
  do {                                                                                                                           \
    auto e = (cuda_call);                                                                                                        \
    if (e != cudaSuccess) Debug() << "[CUDA_ERR](" << __FILE__ << ":" << __LINE__ << ")\t" << cudaGetErrorString(e) << "\n";     \
  } while (0)

#define CURAND_CALL(curand_call)                                                                                                 \
  do {                                                                                                                           \
    auto e = (curand_call);                                                                                                      \
    if (e != CURAND_STATUS_SUCCESS) Debug() << "[CURNAD_ERR](" << __FILE__ << ":" << __LINE__ << ")\n";                          \
  } while (0)

// Get a random number from std::random_device, for seeding pseudo random number generator
inline u32 getRandSeed();

// random array on device
inline void batchRandomGen(u32 *d_array, u32 n);
