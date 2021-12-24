#include "cuckoo_gpu.cuh"

namespace GpuTable {

// test whether x is 0xFFFFFFFF
static __host__ __device__ inline bool empty(u32 x) { return (~x) == 0u; }

Table::Table(u32 cap, u32 t) {
  _m = t, _n = cap;
  for (u32 i = 0; i < _m; i++) _slots[i] = coda::malloc<u32>(_n);
  _backup = coda::malloc<u32>(_n * _m);
  clear();
}

Table::~Table() {
  for (u32 i = 0; i < _m; i++) coda::free(_slots[i]);
  coda::free(_backup);
}

void Table::clear() {
  randomArray(seeds, _m);
  _sz = 0, _inserted = 0;
  for (u32 i = 0; i < _m; i++) coda::fill0xFF(_slots[i], _n);
}

void Table::update(u32 *keys, u32 n) {
  (void)keys;
  (void)n;
}

void Table::query(u32 *keys, u32 *result, u32 n) const {
  (void)keys;
  (void)result;
  (void)n;
}

} // namespace GpuTable
