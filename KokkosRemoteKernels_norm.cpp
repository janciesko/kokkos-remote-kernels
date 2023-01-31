#include "Kokkos_Core.hpp"
#include "KokkosBlas1_nrm2_squared.hpp"
#include "Kokkos_RemoteSpaces.hpp"

#include <mpi.h>
#include <helper.hpp>

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
    
    typedef Kokkos::View<const double*, RemoteSpace_t/*, Kokkos::Device<Kokkos::Cuda, Kokkos::Experimental::NVSHMEMSpace>*/> remote_view_t;
    
/*ORIG*/

  typedef Kokkos::View<
      typename remote_view_t::const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<remote_view_t>::array_layout,
      typename remote_view_t::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      XVector_Internal;
/*  Some mock-ups
    typedef Kokkos::View<
      typename remote_view_t::const_value_type*,
            
     //Kokkos::Device<Kokkos::Cuda, Kokkos::Experimental::NVSHMEMSpace>
      remote_view_t::array_layout,
      /*remote_view_t::memory_space*/
      
      /*remote_view_t::device_type*//* Kokkos::Device<Kokkos::Cuda, Kokkos::Experimental::NVSHMEMSpace>/*,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>*/
      //typename Kokkos::Device<Kokkos::Cuda, Kokkos::Experimental::NVSHMEMSpace>
      //RemoteSpace_t
      /*typename XVector::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>*/// >
   //   XVector_Internal;

    using dst_traits = typename remote_view_t::traits;
    using src_traits = typename XVector_Internal::traits;
    using assign_map = typename Kokkos::Impl::ViewMapping<
          dst_traits, src_traits,
          Kokkos::Experimental::RemoteSpaceSpecializeTag>;
    printf("%i %i %i %i %i %i\n", 
      (int)assign_map::is_assignable,
      (int)assign_map::is_assignable_value_type,
      (int)assign_map::is_assignable_data_type,
      (int)assign_map::is_assignable_dimension,
      (int)assign_map::is_assignable_layout,
      (int)assign_map::is_assignable_space
      );

    remote_view_t myVector("vector", nx_total);
    remote_view_t myVector2 = myVector;
    XVector_Internal internalVector = myVector;
    //XVector_Internal myVector3(myVector);

#if 0
    Kokkos::View<double*, RemoteSpace_t> myVector_2("vector", nx_total);
    Kokkos::parallel_for(Kokkos::RangePolicy(local_range.second, local_range.first),
			 KOKKOS_LAMBDA(const int idx) {     
         myVector(idx) = (double)(myRank + 1);
    });
#endif

    //printf("rank %d: myVector.extent(0)=%d, nx_total=%d\n", myRank, int(myVector.extent(0)), int(nx_total));
  //  const double local_norm = KokkosBlas::nrm2_squared(myVector);
  }
    Kokkos::finalize();
    networking_fin();

}
