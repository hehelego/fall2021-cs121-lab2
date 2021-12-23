#ifndef _CS121_LAB2_CUCKOO_
#define _CS121_LAB2_CUCKOO_

template <u32 HASH_FUNCS = 2, u32 THREADS_PER_BLOCK = 1024> class CuckooHashingTable {
  u32 seed[HASH_FUNCS];
  u32 *_slots, *_keys;
  u32 _cap, _sz;
  u32 _inserted;

public:
  CuckooHashingTable(u32 cap);
  ~CuckooHashingTable();
  void clear();
  void update(u32 *keys, u32 n);
  void query(u32 *keys, u32 *result, u32 n);
};

#endif
