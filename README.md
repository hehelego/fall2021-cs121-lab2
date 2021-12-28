# cs121@fall2021: lab2 cuckoo hashing on GPU with nVidia CUDA

## testing environment setup

hw

- CPU: `Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz`
- GPU: `RTX 2080 Ti` (Turing SM, 11 GB VRAM, PCI-e 3.0x16)
- RAM: 62GB (configuration unknown)

sw

- `ubuntu 18.04`
- `cmake 3.20.2`: `python3 -m pip install -U cmake==3.20.2`
- `fish`
- `CUDA 11.1`

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
