#ifndef _CS121_LAB2_CUCKOO_
#define _CS121_LAB2_CUCKOO_

#include "common.cuh"

namespace GpuTable{
template <u32 M = 2, u32 TPB = 1024> class Table {
  u32 seeds[M];
  u32 *_slots[M], *_backup;
  u32 _n, _sz;
  u32 _inserted;

public:
  Table(u32 cap);
  ~Table();
  void clear();
  void update(u32 *keys, u32 n);
  void query(u32 *keys, u32 *result, u32 n) const;
};
}

#endif
