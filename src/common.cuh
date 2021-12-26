#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>

using u8 = std::uint8_t;
using i8 = std::int8_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;

// For printing debug log
struct Debug {
  std::ostream &stream;
  Debug(std::ostream &stream = std::cerr) : stream(stream) {}
  template <typename T> inline Debug &operator<<(const T &t) {
#ifdef DEBUG
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
// For timing
struct Timer {
  std::chrono::high_resolution_clock::time_point _start, _end;
  Timer() {}
  inline void start() { _start = std::chrono::high_resolution_clock::now(); }
  inline void end() { _end = std::chrono::high_resolution_clock::now(); }
  inline double time_in_second() const {
    auto interval = _end - _start;
    return std::chrono::duration_cast<std::chrono::microseconds>(interval).count() / 1e6;
  }
};

template <typename T>
//  \[ \lfloor \log_2 x \rfloor \]
inline T binaryLength(T x) {
  T r = 0;
  while (x) x >>= 1, r++;
  return r;
}

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
    if (e != cudaSuccess) {                                                                                                      \
      Debug() << "[CUDA_ERR](" << __FILE__ << ":" << __LINE__ << ")\t" << cudaGetErrorString(e) << "\n";                         \
      std::abort();                                                                                                              \
    }                                                                                                                            \
  } while (0)

#define CURAND_CALL(curand_call)                                                                                                 \
  do {                                                                                                                           \
    auto e = (curand_call);                                                                                                      \
    if (e != CURAND_STATUS_SUCCESS) {                                                                                            \
      Debug() << "[CURNAD_ERR](" << __FILE__ << ":" << __LINE__ << ")\n";                                                        \
      std::abort();                                                                                                              \
    }                                                                                                                            \
  } while (0)

// shortcut: fill bytes on CPU
template <typename T> static inline void fill(T *a, u32 n, u32 value) { memset(a, value, sizeof(T) * n); }
// shortcut: fill with zero bytes on CPU
template <typename T> inline void fillZero(T *a, u32 n) { fill(a, n, 0x00u); }
// shortcut: fill with 0xff bytes on CPU
template <typename T> inline void fill0xFF(T *a, u32 n) { fill(a, n, 0xFFu); }

// Get a random number from std::random_device, for seeding pseudo random number generator
inline u32 randomSeed() {
  static std::random_device rand_dev;
  return rand_dev();
}
// random array on device
inline void randomArray(u32 *a, u32 n) {
  std::mt19937 rng(randomSeed());
  for (u32 i = 0; i < n; i++) a[i] = rng();
}

// utilities: wrapper for common cuda calls
namespace coda {
// random array on device
inline void randomArray(u32 *d_array, u32 n) {
  curandGenerator_t rng;
  CURAND_CALL(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rng, randomSeed()));
  CURAND_CALL(curandGenerate(rng, d_array, n));
}
using CopyKind = cudaMemcpyKind;
// host to host. CPU to CPU
const CopyKind H2H = cudaMemcpyHostToHost;
// host to device. CPU to GPU
const CopyKind H2D = cudaMemcpyHostToDevice;
// device to hsot. GPU to CPU
const CopyKind D2H = cudaMemcpyDeviceToHost;
// device to device. GPU to GPU
const CopyKind D2D = cudaMemcpyDeviceToDevice;

// shortcut: copy memory across GPU and CPU
template <typename T> inline void copy(T *dst, const T *src, u32 n, CopyKind kind) {
  CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * n, kind));
}

// shortcut: allocate an array of T with size n on GPU
template <typename T> inline T *malloc(u32 n) {
  T *p = nullptr;
  CUDA_CALL(cudaMalloc((void **)&p, sizeof(T) * n));
  return p;
}

// shortcut: free allocated memory on GPU
template <typename T> inline void free(T *p) { CUDA_CALL(cudaFree(p)); }

// shortcut: fill bytes on GPU
template <typename T> inline void fill(T *d_array, u32 n, u32 value) { CUDA_CALL(cudaMemset(d_array, value, sizeof(T) * n)); }
// shortcut: fill with zero bytes on GPU
template <typename T> inline void fillZero(T *d_array, u32 n) { fill(d_array, n, 0x00u); }
// shortcut: fill with 0xff bytes on GPU
template <typename T> inline void fill0xFF(T *d_array, u32 n) { fill(d_array, n, 0xFFu); }
} // namespace coda

__host__ __device__ u32 xxHash32(u32 seed, u32 value);
