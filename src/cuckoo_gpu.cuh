#pragma once

#include "common.cuh"

namespace GpuTable {
// maximum number of subtables to use
const u32 M_HASH_FUNCS = 5;

class Table {
  u32 seeds[M_HASH_FUNCS];
  u32 *_slots[M_HASH_FUNCS], *_backup;
  u32 _m, _n, _sz;
  u32 _inserted;

public:
  Table(u32 cap, u32 t);
  ~Table();
  void clear();
  void update(u32 *keys, u32 n);
  void query(u32 *keys, u32 *result, u32 n) const;
};
} // namespace GpuTable
