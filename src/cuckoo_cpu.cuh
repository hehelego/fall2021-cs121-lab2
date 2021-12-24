#ifndef _CS121_CUCKOO_CPU_
#define _CS121_CUCKOO_CPU_

#include "common.cuh"
#include <unordered_set>

namespace CpuTable {

// maximum number of subtables to use
const u32 M_HASH_FUNCS=5;

class Table {
  u32 _seeds[M_HASH_FUNCS];
  u32 *_slots[M_HASH_FUNCS], *_backup;
  u32 _m, _n, _threshold, _sz;

  void insertOnce(u32 key);
  bool queryOnce(u32 key) const;
  void rehash();

public:
  Table(u32 cap, u32 t);
  ~Table();
  void clear();
  void update(u32 *keys, u32 n);
  void query(u32 *keys, u32 *result, u32 n) const;
};

class UnorderedMap {
  std::unordered_multiset<u32> table;

public:
  void clear() { table.clear(); }
  void update(u32 *keys, u32 n) {
    for (u32 i = 0; i < n; i++) table.insert(keys[i]);
  }
  void query(u32 *keys, u32 *result, u32 n) const {
    for (u32 i = 0; i < n; i++) result[i] = table.find(keys[i]) != table.end();
  }
};

} // namespace CpuTable

#endif