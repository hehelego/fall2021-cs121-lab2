#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <vector>

// For printing debug log
struct debug {
  std::ostream &stream;
  debug(std::ostream &stream = std::cerr) : stream(stream) {}
  template <typename T> debug &operator<<(const T &t) {
#if (DEBUG == 0)
    stream << t;
#else
    (void)t;
#endif
    return *this;
  }
};
// For reporting performance
struct output {
  std::ostream &stream;
  output(std::ostream &stream = std::cout) : stream(stream) {}
  template <typename T> output &operator<<(const T &t) {
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

// Get a random number from random device, for seeding pseudo random number generator
u32 rand_dev_seed();

// For timing
struct Timer {
  std::chrono::high_resolution_clock::time_point _start, _end;
  Timer() {}
  void start() { _start = std::chrono::high_resolution_clock::now(); }
  void end() { _end = std::chrono::high_resolution_clock::now(); }
  double time_in_second() const {
    auto interval = _end - _start;
    return std::chrono::duration_cast<std::chrono::microseconds>(interval).count() / 1000.0;
  }
};
