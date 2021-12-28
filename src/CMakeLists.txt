###
### compiling&linking options
###

# Enable cuda device debugging for Debug build.
# Enable optimization for Release build.
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(NOTICE "[build type] debug mode on")
    add_compile_options(
        -O0
        -G
        -g
        -lineinfo
        -Xcompiler=-rdynamic
    )
    add_link_options(
        -g
        -rdynamic
    )
    add_definitions(-D DEBUG)
else()
    message(NOTICE "[build type] debug mode off")
endif()

###
### Dependencies: CUDA::curand
###

find_package(CUDAToolkit)

###
### Libraries
###

# Library for utilities. Depends on CUDA::curand
add_library(common common.cu)
target_link_libraries(common
    CUDA::curand
)

# Library for CuckooHashTable on GPU(device)
add_library(cuckoo_gpu cuckoo_gpu.cu)
target_link_libraries(cuckoo_gpu common)

# Library for CuckooHashTable on CPU(host)
add_library(cuckoo_cpu cuckoo_cpu.cu)
target_link_libraries(cuckoo_cpu common)

###
### Executables
###

# Executable target for performance benchmarking
add_executable(bench bench.cu)
target_link_libraries(bench
    common
    cuckoo_gpu
    cuckoo_cpu
)
# Executable target for correctness verification
add_executable(verify verify.cu)
target_link_libraries(verify
    common
    cuckoo_gpu
    cuckoo_cpu
)
