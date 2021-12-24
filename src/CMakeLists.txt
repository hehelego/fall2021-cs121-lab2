if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(NOTICE "[build type] debug mode on")
    add_compile_options(-g)
    add_link_options(-g)
    add_definitions(-D DEBUG)
else()
    message(NOTICE "[build type] debug mode off")
endif()

find_package(CUDAToolkit)
add_library(cuckoo_gpu cuckoo_gpu.cu)
add_library(cuckoo_cpu cuckoo_cpu.cu)
add_library(common common.cu)
target_link_libraries(common CUDA::curand)

add_executable(run main.cu)
target_link_libraries(run
    common
    cuckoo_gpu
    cuckoo_cpu
)
