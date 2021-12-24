#include "cuckoo_gpu.cuh"

namespace GpuTable
{

// test whether x is 0xFFFFFFFF
static __host__ __device__ inline bool empty(u32 x) { return (~x) == 0u; }

template <u32 M, u32 TPB> Table<M, TPB>::Table(u32 cap) {
  _n = cap;
  for(u32 i=0;i<M;i++) _slots[i] = coda::malloc<u32>(_n);
  _backup = coda::malloc<u32>(_n*M);
  clear();
}

template <u32 M, u32 TPB> Table<M, TPB>::~Table() {
  for(u32 i=0;i<M;i++) coda::free(_slots[i]);
  coda::free(_backup);
}

template <u32 M, u32 TPB> void Table<M, TPB>::clear() {
  randomArray(seeds, M);
  _sz = 0, _inserted = 0;
  for (u32 i = 0; i < M; i++) coda::fill0xFF(_slots[i], _n);
}

template <u32 M, u32 TPB>
void Table<M, TPB>::update(u32 *keys, u32 n) {
  (void)keys;
  (void)n;
}

template <u32 M, u32 TPB>
void Table<M, TPB>::query(u32 *keys, u32 *result, u32 n) const {
  (void)keys;
  (void)result;
  (void)n;
}
  
} // namespace GpuTable
