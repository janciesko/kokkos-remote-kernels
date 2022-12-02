cmake_minimum_required(VERSION 3.13)
project(KokkosRemoteKernels LANGUAGES CXX)

include(CTest)

find_package(KokkosKernels REQUIRED)
find_package(KokkosRemote REQUIRED)

set(HEADER_DIRS)
list(APPEND HEADER_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(KokkosRemoteKernels_MatVec KokkosRemoteKernels_norm.cpp)
target_include_directories(KokkosRemoteKernels_MatVec PRIVATE ${HEADER_DIRS})
target_link_libraries(KokkosRemoteKernels_MatVec PRIVATE Kokkos::kokkoskernels Kokkos::kokkosremote)
