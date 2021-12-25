#include "cuckoo_cpu.cuh"

namespace CpuTable {
// test whether x is 0xFFFFFFFF
static inline bool empty(u32 x) { return (~x) == 0u; }

Table::Table(u32 cap, u32 t) {
  _n = cap, _m = t;
  _threshold = binaryLength(_n) * 4;
  for (u32 i = 0; i < _m; i++) _slots[i] = new u32[_n];
  clear();
}

Table::~Table() {
  for (u32 i = 0; i < _m; i++) delete[] _slots[i];
}

void Table::clear() {
  while (true) {
    randomArray(_seeds, _m);
    bool ok = true;
    for (u32 i = 0; i < _m; i++) {
      for (u32 j = 0; j < i; j++) ok &= (_seeds[i] != _seeds[j]);
    }
    if (!ok) {
      randomArray(_seeds, _m);
    } else {
      break;
    }
  }
  _sz = 0;
  for (u32 i = 0; i < _m; i++) fill0xFF(_slots[i], _n);
}

void Table::rehash() {
  Debug() << "CPU TABLE: rehash\n";

  u32 *backup[M_HASH_FUNCS];
  for (u32 i = 0; i < _m; i++) {
    backup[i] = _slots[i];
    _slots[i] = new u32[_n];
  }
  for (u32 i = 0; i < _m; i++) {
    for (u32 j = 0; j < _n; j++) insertOnce(backup[i][j]);
  }
  clear();
  for (u32 i = 0; i < _m; i++) delete[] backup[i];
}
void Table::insertOnce(u32 key) {
  if (empty(key)) return;

  bool ok = false;
  u32 cnt = 0;
  do {
    if (cnt >= _threshold) {
      rehash();
      cnt = 0;
    }
    for (u32 i = 0; i < _m; i++) {
      u32 slot = xxHash32(_seeds[i], key) % _n;
      std::swap(key, _slots[i][slot]);
      if (empty(key) || key == _slots[i][slot]) {
        ok = true;
        break;
      } else {
        cnt++;
      }
    }
  } while (!ok);
  _sz++;
}
bool Table::queryOnce(u32 key) const {
  if (empty(key)) return false;

  u32 cnt = 0;
  for (u32 i = 0; i < _m; i++) {
    u32 slot = xxHash32(_seeds[i], key) % _n;
    cnt += _slots[i][slot] == key;
  }
  return cnt > 0;
}

void Table::update(u32 *keys, u32 n) {
  for (u32 i = 0; i < n; i++) insertOnce(keys[i]);
}
void Table::query(u32 *keys, u32 *result, u32 n) const {
  for (u32 i = 0; i < n; i++) result[i] = queryOnce(keys[i]);
}

//////////////////////////////
void UnorderedMap::clear() { table.clear(); }
void UnorderedMap::update(u32 *keys, u32 n) {
  for (u32 i = 0; i < n; i++) {
    if (!empty(keys[i])) table.insert(keys[i]);
  }
}
void UnorderedMap::query(u32 *keys, u32 *result, u32 n) const {
  for (u32 i = 0; i < n; i++) result[i] = table.find(keys[i]) != table.end();
}

} // namespace CpuTable
