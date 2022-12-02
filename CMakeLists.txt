cmake_minimum_required(VERSION 3.13)
project(KokkosRemoteKernels LANGUAGES CXX)

include(CTest)

find_package(KokkosKernels REQUIRED)
find_package(KokkosRemote REQUIRED)

add_executable(KokkosRemoteKernels_MatVec KokkosRemoteKernels_norm.cpp)
target_link_libraries(KokkosRemoteKernels_MatVec PRIVATE Kokkos::kokkoskernels Kokkos::kokkosremote)
