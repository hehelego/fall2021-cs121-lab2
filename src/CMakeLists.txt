if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(NOTICE "[build type] debug mode on")
    add_compile_options(
        -O0
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

find_package(CUDAToolkit)
add_library(common common.cu)
add_library(cuckoo_gpu cuckoo_gpu.cu)
add_library(cuckoo_cpu cuckoo_cpu.cu)
target_link_libraries(common CUDA::curand)
target_link_libraries(cuckoo_cpu common)
target_link_libraries(cuckoo_gpu common)

add_executable(run main.cu)
target_link_libraries(run
    common
    cuckoo_gpu
    cuckoo_cpu
)