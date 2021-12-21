#include "common.hpp"

u32 rand_dev_seed() {
  std::random_device rand_dev;
  return rand_dev();
}
