#include "common.cuh"
#include "cuckoo.cuh"

// test whether x is 0xFFFFFFFF
static __host__ __device__ inline bool empty(u32 x) { return (~x) == 0u; }

template <u32 HASH_FUNCS, u32 THREADS_PER_BLOCK> CuckooHashingTable<HASH_FUNCS, THREADS_PER_BLOCK>::CuckooHashingTable(u32 cap) {
  _slots = coda::malloc<u32>(cap), _keys = coda::malloc<u32>(cap);
  _cap = cap;
  clear();
}

template <u32 HASH_FUNCS, u32 THREADS_PER_BLOCK> CuckooHashingTable<HASH_FUNCS, THREADS_PER_BLOCK>::~CuckooHashingTable() {
  coda::free(_slots), coda::free(_keys);
}

template <u32 HASH_FUNCS, u32 THREADS_PER_BLOCK> void CuckooHashingTable<HASH_FUNCS, THREADS_PER_BLOCK>::clear() {
  for (u32 i = 0; i < HASH_FUNCS; i++) seed[i] = getRandSeed();

  _sz = 0, _inserted = 0;

  coda::fill0xFF(_slots, _cap), coda::fill0xFF(_keys, _cap);
}

template <u32 HASH_FUNCS, u32 THREADS_PER_BLOCK>
void CuckooHashingTable<HASH_FUNCS, THREADS_PER_BLOCK>::update(u32 *keys, u32 n) {
  (void)keys;
  (void)n;
}

template <u32 HASH_FUNCS, u32 THREADS_PER_BLOCK>
void CuckooHashingTable<HASH_FUNCS, THREADS_PER_BLOCK>::query(u32 *keys, u32 *result, u32 n) {
  (void)keys;
  (void)result;
  (void)n;
}
