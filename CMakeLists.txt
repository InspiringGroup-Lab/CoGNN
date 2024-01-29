# # SGX
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/SGX-Worker/cmake)
# find_package(SGX REQUIRED)

# # emp
# find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
# include(${CMAKE_FOLDER}/cmake/emp-base.cmake)
# include(${CMAKE_FOLDER}/cmake/enable_rdseed.cmake)
# # include(${CMAKE_FOLDER}/cmake/enable_float.cmake)

# find_package(emp-sh2pc REQUIRED)
# include_directories(${EMP-SH2PC_INCLUDE_DIRS})

# # ophelib
# list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/SSHE-Worker/cmake/OPHE/cmake")
# find_package(NTL REQUIRED)
# find_package(GMP REQUIRED)
# find_package(GMPXX REQUIRED)
# find_package(OPHELib REQUIRED)
# find_package(Threads REQUIRED)
# find_package(OpenMP)

include_directories(include)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

# add_subdirectory(algo_kernels/edge_centric/ss-pagerank)

# add_subdirectory(algo_kernels/edge_centric/ss-bfs)

# add_subdirectory(algo_kernels/edge_centric/ss-cc)

# add_subdirectory(algo_kernels/edge_centric/ss-sssp)

add_subdirectory(algo_kernels/edge_centric/ss-gcn)
add_subdirectory(algo_kernels/edge_centric/optimize-gcn)
add_subdirectory(algo_kernels/edge_centric/optimize-gcn-inference)
add_subdirectory(algo_kernels/edge_centric/original-gcn)