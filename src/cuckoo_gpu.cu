#include "common.cuh"
#include "cuckoo_gpu.cuh"

#include <utility>

namespace GpuTable {
// test whether x is 0xFFFFFFFF
static __device__ inline bool empty(u32 x) { return (~x) == 0u; }
static const u32 EMPTY = (~0u);

static __global__ void rollBackKernel(std::pair<u32, u32> *result, u32 n, u32 **slots) {
  u32 i = threadIdx.x + blockIdx.x * blockDim.x;
  u32 final_table = EMPTY, final_slot = EMPTY;
  if (i < n) {
    final_table = result[i].first;
    final_slot = result[i].second;
    if (!empty(final_table) && !empty(final_slot)) slots[final_table][final_slot] = EMPTY;
  }
}
static __global__ void updateKernel(u32 *keys, std::pair<u32, u32> *result, u32 n, u32 cap, u32 **slots, const u32 *const seeds,
                                    u32 m, u32 threshold, u32 *counter) {
  u32 i = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ u32 cachedSeeds[M_HASH_FUNCS];
  if (i < m) cachedSeeds[i] = seeds[i];
  __syncthreads();

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
      atomicAdd(counter, 1);
      result[i].first = EMPTY;
      result[i].second = EMPTY;
    } else {
      result[i].first = final_table;
      result[i].second = final_slot;
    }
  }
}
static __global__ void queryKernel(const u32 *keys, u32 *result, u32 n, u32 cap, u32 **const slots, const u32 *const seeds,
                                   u32 m) {
  u32 i = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ u32 cachedSeeds[M_HASH_FUNCS];
  if (i < m) cachedSeeds[i] = seeds[i];
  __syncthreads();

  u32 i = threadIdx.x + blockIdx.x * blockDim.x;
  u32 key = 0;
  if (i < n && !empty(keys[i])) {
    key = keys[i];
    for (u32 j = 0; j < m; j++) {
      u32 slot = xxHash32(cachedSeeds[j], key) % cap;
      result[i] += slots[j][slot] == key;
    }
  }
}
static inline u32 queryMaxThreadsPerBlock() {
  cudaDeviceProp dp;
  CUDA_CALL(cudaGetDeviceProperties(&dp, 0));
  return dp.maxThreadsPerBlock;
}
Table::Table(u32 cap, u32 t) : THREADS_PER_BLOCK(queryMaxThreadsPerBlock()) {
  _n = cap, _m = t;
  _threshold = binaryLength(_n) * 4;
  _slots = coda::malloc<u32 *>(_m);
  for (u32 i = 0; i < _m; i++) _slotsHost[i] = coda::malloc<u32>(_n);
  coda::copy(_slots, _slotsHost, _m, coda::H2D);
  _seeds = coda::malloc<u32>(M_HASH_FUNCS);
  clear();
}

Table::~Table() {
  coda::copy(_slotsHost, _slots, _m, coda::D2H);
  for (u32 i = 0; i < _m; i++) coda::free(_slotsHost[i]);
  coda::free(_slots);
  coda::free(_seeds);
}

void Table::clear() {
  coda::randomArray(_seeds, _m);
  _sz = 0;
  coda::copy(_slotsHost, _slots, _m, coda::D2H);
  for (u32 i = 0; i < _m; i++) coda::fill0xFF(_slotsHost[i], _n);
}

void Table::rehash() {
  Debug() << "GPU TABLE: rehash\n";
  u32 *backup[M_HASH_FUNCS] coda::copy(_slotsHost, _slots, _m, coda::D2H);
  for (u32 i = 0; i < _m; i++) {
    backup[i] = _slotsHost[i];
    _slotsHost[i] = coda::malloc<u32>(_n);
  }
  coda::copy(_slots, _slotsHost, _m, coda::H2D);
  clear();
  for (u32 i = 0; i < _m; i++) update(backup[i], _n);
  for (u32 i = 0; i < _m; i++) coda::free(backup[i]);
}

void Table::update(u32 *keys, u32 n) {
  u32 blocks = div_ceil(n, THREADS_PER_BLOCK);
  u32 *d_counter = coda::malloc<u32>(1), counter = 0;
  auto result = coda::malloc<std::pair<u32, u32>>(n);
  while (true) {
    coda::fillZero(d_counter, 1);
    updateKernel<<<blocks, THREADS_PER_BLOCK>>>(keys, result, n, _n, _slots, _seeds, _m, _threshold, d_counter);
    coda::copy(&counter, d_counter, 1, coda::D2H);
    if (counter > 0) {
      rollBackKernel<<<blocks, THREADS_PER_BLOCK>>>(result, n, _slots);
      rehash();
    } else {
      break;
    }
  }
  coda::free(d_counter), coda::free(result);

  _sz += n;
}
void Table::query(u32 *keys, u32 *result, u32 n) const {
  u32 blocks = div_ceil(n, THREADS_PER_BLOCK);
  coda::fillZero(result, n);
  queryKernel<<<blocks, THREADS_PER_BLOCK>>>(keys, result, n, _n, _slots, _seeds, _m);
}

} // namespace GpuTable
