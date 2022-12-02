# Kokkos Remote Kernels

This repository contains sample codes that shows how Kokkos remote space and Kokkos Kernels can be used to implement distribute linear algebra kernels using PGAS as the primary communication layer.

### Building Kokkos Remote Kernels

First you will need to build Kokkos, Kokkos Kernels and Kokkos remote spaces using a consistent set of compilers.
Once all are installed you can configure using CMake as follows

```
cmake -S path/to/kokkos_remote_kernels/source \
    -B path/to/the/build/directory \
    -D CMAKE_CXX_COMPILER="same compiler as that used with Kokkos remote spaces" \
    -D KokkosKernels_ROOT:PATH="path/to/KokkosKernels/installation" \
    -D KokkosRemote_ROOT:PATH="path/to/Kokkos_Remote_Spaces/installation"
```
