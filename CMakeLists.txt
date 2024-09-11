cmake_minimum_required(VERSION 3.8)
project(TileSpGEMM LANGUAGES C CXX CUDA)

# set c++ standard and find packages
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)
find_package(CUDAToolkit REQUIRED)
set(Headers
)

string(APPEND CMCAKE_C_FLAGS " -fopenmp ")
string(APPEND CMCAKE_CXX_FLAGS " -fopenmp ")

# AcSpGEMM Library
add_library(tileSpGEMM
    src/utils.cpp
    src/spgemm-cpu.cpp
    src/spgemm_serialref_spa_new.cpp
    src/mmio_highlevel.cpp
    src/csr2tile.cpp 
    src/utils_cuda_scan.cu
    src/spgemm_nsparse_kernel.cu
    src/tilespgemm-cuda.cu
    ${Headers}
)

find_package(OpenMP REQUIRED)
target_link_libraries(tileSpGEMM PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(tileSpGEMM PUBLIC include)
target_include_directories(tileSpGEMM PUBLIC include/external)
target_include_directories(tileSpGEMM PUBLIC externals)

string(APPEND CMAKE_CUDA_FLAGS " -I${CMAKE_CURRENT_SOURCE_DIR}/include ")
string(APPEND CMAKE_CUDA_FLAGS " -I${CMAKE_CURRENT_SOURCE_DIR}/include/external ")
string(APPEND CMAKE_CUDA_FLAGS " -I${CMAKE_CURRENT_SOURCE_DIR}/externals ")
target_link_libraries(tileSpGEMM PUBLIC CUDA::cudart CUDA::cusparse)

add_executable(main src/main.cu)
target_link_libraries(main tileSpGEMM)


# Comparison implementations for cuSparse
# set(COMP_SOURCES "externals/cusparse/source/cuSparseMultiply.cu")

# add_library(comp_implementations
# 				${COMP_SOURCES}
# )

# # Executable to run a single test
# add_executable(HostTest 
#                 source/main.cpp
# )

# # Executable to perform a complete testrun
# add_executable(performTestCase
# 				source/performTestCase.cpp
# )

# # Executable to test bit stability
# add_executable(checkBitStability
# 				source/checkBitStability.cpp
# )


# Link Libraries
# target_link_libraries(HostTest acSpGEMM comp_implementations ${CUDA_cudart_static_LIBRARY} cusparse )
# target_link_libraries(performTestCase acSpGEMM comp_implementations ${CUDA_cudart_static_LIBRARY} cusparse ${GCC_COVERAGE_LINK_FLAGS} )
# target_link_libraries(checkBitStability acSpGEMM comp_implementations ${CUDA_cudart_static_LIBRARY} cusparse ${GCC_COVERAGE_LINK_FLAGS} )

# target_link_libraries(HostTest acSpGEMM CUDA::cudart CUDA::cusparse)
# target_link_libraries(performTestCase acSpGEMM ${CUDA_cudart_static_LIBRARY} CUDA::cusparse ${GCC_COVERAGE_LINK_FLAGS} )
# target_link_libraries(checkBitStability acSpGEMM ${CUDA_cudart_static_LIBRARY} CUDA::cusparse ${GCC_COVERAGE_LINK_FLAGS} )
