cmake_minimum_required(VERSION 3.10)
project(CS121_LAB2_CuckooHashing)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(src bin)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(NOTICE "[build type] debug mode on")
    add_compile_options(-g)
    add_link_options(-g)
    add_definitions(-D DEBUG)
else()
    message(NOTICE "[build type] debug mode off")
endif()

find_package(CUDAToolkit)
add_library(cuckoo_gpu src/cuckoo_gpu.cu)
add_library(cuckoo_cpu src/cuckoo_cpu.cu)
add_library(common src/common.cu)
target_link_libraries(common CUDA::curand)

add_executable(run src/main.cu)
target_link_libraries(run
    common
    cuckoo_gpu
    cuckoo_cpu
)
