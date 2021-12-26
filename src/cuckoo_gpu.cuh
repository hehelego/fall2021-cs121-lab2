#pragma once

#include "common.cuh"

namespace GpuTable {

// maximum number of subtables to use
const u32 M_HASH_FUNCS = 5;
// const u32 THREADS_PER_BLOCK = 1024;
const u32 THREADS_PER_BLOCK = 8;

class Table {
  u32 *_seeds;
  u32 *_slotsHost[M_HASH_FUNCS];
  u32 **_slots;
  u32 _m, _n, _threshold, _sz;

  void rehash();

public:
  Table(u32 cap, u32 t);
  ~Table();
  void clear();
  void update(u32 *keys, u32 n);
  void query(u32 *keys, u32 *result, u32 n) const;
};

} // namespace GpuTable
