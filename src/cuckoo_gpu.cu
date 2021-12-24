#include "cuckoo_gpu.cuh"

#include <utility>

namespace GpuTable {
// test whether x is 0xFFFFFFFF
static __device__ inline bool empty(u32 x) { return (~x) == 0u; }
static const u32 EMPTY = (~0u);
static __constant__ u32 cachedSeeds[M_HASH_FUNCS];

static __global__ inline void rollBackKernel(std::pair<u32, u32> *result, u32 n, u32 *slots[M_HASH_FUNCS]) {
  u32 i = threadIdx.x + blockIdx.x * blockDim.x;
  u32 final_table = EMPTY, final_slot = EMPTY;
  if (i < n) {
    final_table = result[i].first;
    final_slot = result[i].second;
    if (!empty(final_table) && !empty(final_slot)) { slots[final_table][final_slot] = EMPTY; }
  }
}
static __global__ inline void updateKernel(u32 *keys, std::pair<u32, u32> *result, u32 n, u32 cap, u32 *slots[M_HASH_FUNCS],
                                           u32 m, u32 threshold, u32 &counter) {
  u32 i = threadIdx.x + blockIdx.x * blockDim.x;
  u32 key = 0, final_table = EMPTY, final_slot = EMPTY;
  if (i < n) {
    key = keys[i];
    for (u32 j = 0; !empty(key) && j < threshold; j++) {
      u32 jj = j % m;
      u32 slot = xxHash32(cachedSeeds[jj], key) % cap;
      final_table = jj, final_slot = slot;
      key = atomicExch(&slots[jj][slot], key);
    }
    if (!empty(key)) {
      atomicAdd(&counter, 1);
      result[i].first = EMPTY;
      result[i].second = EMPTY;
    } else {
      result[i].first = final_table;
      result[i].second = final_slot;
    }
  }
}
static __global__ inline void queryKernel(const u32 *keys, u32 *result, u32 n, u32 cap, u32 *const slots[M_HASH_FUNCS], u32 m) {
  u32 i = threadIdx.x + blockIdx.x * blockDim.x;
  u32 key = 0;
  if (i < n && !empty(keys[i])) {
    key = keys[i];
    for (u32 j = 0; j < m; j++) {
      u32 slot = xxHash32(cachedSeeds[j], key) % cap;
      result[slot] += slots[j][slot] == key;
    }
  }
}

Table::Table(u32 cap, u32 t) {
  _n = cap, _m = t;
  _threshold = binaryLength(_n) * 4;
  for (u32 i = 0; i < _m; i++) _slots[i] = coda::malloc<u32>(_n);
  clear();
}

Table::~Table() {
  for (u32 i = 0; i < _m; i++) coda::free(_slots[i]);
}

void Table::clear() {
  randomArray(_seeds, _m);
  _sz = 0;
  for (u32 i = 0; i < _m; i++) coda::fill0xFF(_slots[i], _n);
}

void Table::rehash() {
  Debug() << "GPU TABLE: rehash\n";
  u32 *backup[M_HASH_FUNCS];
  for (u32 i = 0; i < _m; i++) {
    backup[i] = _slots[i];
    _slots[i] = coda::malloc<u32>(_n);
  }
  clear();
  for (u32 i = 0; i < _m; i++) update(backup[i], _n);
  for (u32 i = 0; i < _m; i++) coda::free(backup[i]);
}

void Table::update(u32 *keys, u32 n) {
  u32 counter = 0;
  u32 blocks = div_ceil(n, THREADS_PER_BLOCK);
  auto result = coda::malloc<std::pair<u32, u32>>(n);
  while (true) {
    coda::copyConst(cachedSeeds, _seeds, _m);
    updateKernel<<<blocks, THREADS_PER_BLOCK>>>(keys, result, n, _n, _slots, _m, _threshold, counter);
    if (counter > 0) {
      rollBackKernel<<<blocks, THREADS_PER_BLOCK>>>(result, n, _slots);
      rehash();
      counter = 0;
    } else {
      break;
    }
  }
  coda::free(result);
  _sz += n;
}
void Table::query(u32 *keys, u32 *result, u32 n) const {
  coda::copyConst(cachedSeeds, _seeds, _m);
  coda::fillZero(result, n);
  u32 blocks = div_ceil(n, THREADS_PER_BLOCK);
  queryKernel<<<blocks, THREADS_PER_BLOCK>>>(keys, result, n, _n, _slots, _m);
}

} // namespace GpuTable
