#include "cuckoo_cpu.cuh"

namespace CpuTable {

// test whether x is 0xFFFFFFFF
static inline bool empty(u32 x) { return (~x) == 0u; }

Table::Table(u32 cap, u32 t) {
  _n = cap, _m = t;
  _threshold = binaryLength(_n) * 4;
  for (u32 i = 0; i < _m; i++) _slots[i] = new u32[_n];
  _backup = new u32[_n * _m];
  clear();
}

Table::~Table() {
  for (u32 i = 0; i < _m; i++) delete[] _slots[i];
  delete[] _backup;
}

void Table::clear() {
  randomArray(_seeds, _m);
  _sz = 0;
  for (u32 i = 0; i < _m; i++) fill0xFF(_slots[i], _n);
}

void Table::rehash() {
  Debug() << "CPU TABLE: rehash\n";
  u32 sz = _sz;
  clear();
  for (u32 i = 0; i < sz; i++) insertOnce(_backup[i]);
}
void Table::insertOnce(u32 key) {
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
      if (empty(key)) {
        ok = true;
        break;
      } else {
        cnt++;
      }
    }
  } while (!ok);
  _backup[_sz++] = key;
}
bool Table::queryOnce(u32 key) const {
  u32 cnt = 0;
  for (u32 i = 0; i < _m; i++) {
    u32 slot = xxHash32(_seeds[i], key) % _n;
    cnt += _slots[i][slot] == key;
  }
  return cnt > 0;
}

void Table::update(u32 *keys, u32 n) {
  clear();
  for (u32 i = 0; i < n; i++) insertOnce(keys[i]);
}
void Table::query(u32 *keys, u32 *result, u32 n) const {
  for (u32 i = 0; i < n; i++) result[i] = queryOnce(keys[i]);
}

} // namespace CpuTable
