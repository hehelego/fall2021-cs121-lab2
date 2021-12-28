#include "common.cuh"
#include "cuckoo_gpu.cuh"

#include <fstream>
#include <functional>
#include <iostream>

// key file. query file. output file
void test(const std::string &kfile, const std::string &qfile, const std::string &ofile) {
  std::ifstream ink(kfile), inq(qfile);

  u32 n = 0, m = 0;
  ink >> n, inq >> m;
  u32 *keys = new u32[n], *qrys = new u32[m], *res = new u32[m];
  for (u32 i = 0; i < n; i++) ink >> keys[i];
  for (u32 i = 0; i < m; i++) inq >> qrys[i];

  u32 *d_keys = coda::malloc<u32>(n), *d_qrys = coda::malloc<u32>(m), *d_res = coda::malloc<u32>(m);
  GpuTable::Table table(n * 2, 3);
  table.update(d_keys, n);
  table.query(d_qrys, d_res, m);
  coda::copy(res, d_res, m, coda::D2H);

  std::ofstream out(ofile);
  for (u32 i = 0; i < m; i++) out << d_res[i] << " ";
}

int main(int argc, const char *argv[]) {
  if (argc < 4) {
    Output()
        << "Usage:"
        << "\t" << argv[0] << "key_file query_file result_file"
        << "\n\t"
        << "-key file: The number of keys on the first line. List all the keys on the second line, separated by space."
        << "**The keys should be unique**."
        << "\n\t"
        << "-query file: The number of queries on the first line. List all the queries on the second line, separated by space."
        << "\n\t"
        << "-result file: A line of space separated 0/1 sequence on a single line. (0,not found)/(1,found)"
        << "\n";
    return 1;
  }
  const std::string key_file(argv[1]);
  const std::string qry_file(argv[2]);
  const std::string res_file(argv[3]);

  test(key_file, res_file, res_file);
  return 0;
}