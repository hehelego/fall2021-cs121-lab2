# cs121@fall2021: lab2 cuckoo hashing on GPU with nVidia CUDA

## directory structure

- `README.md`: this file.
- `.git`: git repository.
   `master` brench is the main brench where all the testing outcomes and report can be found.  
   `cuckoo-with-stash` brench contains the implementation of Cuckoo hash table with a small stash.
- `all.fish`: a fish shell script to build the source code. **Before testing, please fill the correct GPU SM arch code in the script.**
- `src/`: source code directory.
- `bin/`: directory that stores all the libraries and binary exectuables.
- `copy-all.fish copy-src.fish`: fish shell scripts to copy this repository to ShanghaiTech AI cluster using `rsync` via `ssh`.
- `report/`: final report. LaTeX source code, bibliography and PDF output are in that directory.

## testing setup

### main hardware information

- CPU: `Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz`
- GPU: `RTX 2080 Ti` (Turing SM, 11 GB VRAM, PCI-e 3.0x16)
- RAM: 62GB (configuration unknown)

### main software version and dependency information

- `ubuntu 18.04`
- `cmake 3.20.2`: use `python3 -m pip install -U cmake==3.20.2` to fetch and install pre-built binary exectuable of cmake from PYPI.
- `CUDA 11.1`: `curand` in CUDA toolkit is required.
- `fish`

## usage

```fish
# please update the $GPU_SM_ARCH correctly before running the script
./all.fish

# verify the correctness. (compared with cuckoo hashing table on CPU and C++ STL std::unordered_set)
./bin/verify

# run the four specified tasks. result printed in stdout
./bin/bench
# Time(xxx.yyy): the task takes `xxx.yyy` microseconds (1ms = 10^-6 second)
# InvalidTime: Too much rehash calls invoked, we determine that the insertion can not be done.

# run Input, insert, query, Output.
# with user specified key file, query file and output file.
./bin/run key_file query_file output_file
# key_file:
#    Path to a file that contains all the keys.
#    A unsigned integer on the first line indicating the number of keys in this file.
#    A sequence of space separated unsigned integers: the keys to be inserted.
#    The keys should be unique.
#
# query_file:
#    Path to a file that contains all the queries.
#    A unsigned integer on the first line indicating the number of queries in this file.
#    A sequence of space separated unsigned integers.
#    Duplicated keys are allowed.
#
# output_file:
#    Path to a file.
#    The output will be written to that file.
#    The result is encoded in a space separated 0/1 string.
#    0 means that the corresponding key does not exists in the key file.
#    1 means that the corresponding key can be found in the key file.
```

## sample benchmark result

```plaintext
benchmark 1: insertion
10    Time(41.600000)
11    Time(31.800000)
12    Time(35.400000)
13    Time(37.800000)
14    Time(38.200000)
15    Time(51.200000)
16    Time(82.600000)
17    Time(136.400000)
18    Time(1364.000000)
19    Time(944.400000)
20    Time(1423.600000)
21    Time(2406.200000)
22    Time(4403.000000)
23    Time(8979.000000)
24    Time(19222.600000)
benchmark 2: query
0               Time(579.000000)
1               Time(577.000000)
2               Time(580.000000)
3               Time(577.800000)
4               Time(578.400000)
5               Time(717.200000)
6               Time(712.400000)
7               Time(711.000000)
8               Time(710.200000)
9               Time(711.400000)
10              Time(710.800000)
benchmark 3: insertion with high load factor
1               Time(91473.000000)
2               Time(52895.400000)
3               Time(32446.200000)
4               Time(85283.200000)
5               Time(51899.400000)
6               Time(52622.800000)
7               Time(31932.400000)
8               Time(25061.800000)
9               Time(38706.600000)
10              Time(59105.000000)
1.01            InvalidTime
1.02            InvalidTime
1.05            InvalidTime
benchmark 4: eviction chain limit
1.000000                Time(24727.800000)
1.500000                Time(25054.400000)
2.000000                Time(32479.400000)
2.500000                Time(25055.600000)
3.000000                Time(25241.400000)
3.500000                Time(25129.800000)
4.000000                Time(25703.400000)
4.500000                Time(32322.800000)
5.000000                Time(25097.400000)
5.500000                Time(25116.400000)
6.000000                Time(25053.400000)
6.500000                Time(25135.800000)
7.000000                Time(25068.800000)
7.500000                Time(25050.200000)
```
