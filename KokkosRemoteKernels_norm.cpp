#include "Kokkos_Core.hpp"
#include "KokkosBlas1_nrm2_squared.hpp"
#include "Kokkos_RemoteSpaces.hpp"

#include <mpi.h>

using RemoteSpace_t  = Kokkos::Experimental::DefaultRemoteMemorySpace;

int main(int argc, char *argv[]) {
  // MPI
  int myRank, numRanks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

#ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_init();
#endif
#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm      = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  Kokkos::initialize(argc, argv);
  {
    printf("Hello from rank %d!\n", myRank);

    const int nx_per_rank = 16;
    const int nx_total    = nx_per_rank*numRanks;

    Kokkos::View<double*, RemoteSpace_t> myVector("vector", nx_total);
    Kokkos::parallel_for(Kokkos::RangePolicy(myRank*nx_per_rank, (myRank + 1)*nx_per_rank),
			 KOKKOS_LAMBDA(const int idx) {myVector(idx) = (double)(myRank + 1);});

    printf("rank %d: myVector.extent(0)=%d, nx_total=%d\n", myRank, int(myVector.extent(0)), int(nx_total));

    const double norm = KokkosBlas::nrm2_squared(myVector);
  }
  Kokkos::finalize();
}
