#include "Kokkos_Core.hpp"
#include "KokkosBlas1_nrm2_squared.hpp"
#include "Kokkos_RemoteSpaces.hpp"

#include <mpi.h>
#include <networking_code.hpp>

using RemoteSpace_t  = Kokkos::Experimental::DefaultRemoteMemorySpace;
#define SIZE 1024

int main(int argc, char *argv[]) {

  networking_init(argc, argv);

  Kokkos::initialize(argc, argv);
  {
    int myRank;
    int numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    
    printf("Hello from rank %d!\n", myRank);
    int nx_total     = SIZE;
    auto local_range = Kokkos::Experimental::get_range(nx_total, myRank);
    
    Kokkos::View<double*, RemoteSpace_t> myVector("vector", nx_total);
    Kokkos::parallel_for(Kokkos::RangePolicy(local_range.second, local_range.first),
			 KOKKOS_LAMBDA(const int idx) {     
         myVector(idx) = (double)(myRank + 1);
    });

    printf("rank %d: myVector.extent(0)=%d, nx_total=%d\n", myRank, int(myVector.extent(0)), int(nx_total));
    const double norm = KokkosBlas::nrm2_squared(myVector);
  }
    Kokkos::finalize();
    networking_fin();

}
