#include "Kokkos_Core.hpp"
#include "KokkosBlas1_nrm2_squared.hpp"
#include "Kokkos_RemoteSpaces.hpp"

#include <mpi.h>
#include <helper.hpp>

using RemoteSpace_t  = Kokkos::Experimental::DefaultRemoteMemorySpace;
#define SIZE 1024

template <class DstTraits, class SrcTraits>
void print_view_type_accessiblity(DstTraits dV, SrcTraits sV)
{
    using dst_traits = typename DstTraits::traits;
    using src_traits = typename SrcTraits::traits;

    using assign_map = typename Kokkos::Impl::ViewMapping<
          dst_traits, src_traits,
          typename dst_traits::specialize>;

    printf("%i %i %i %i %i %i\n", 
      (int)assign_map::is_assignable,
      (int)assign_map::is_assignable_value_type,
      (int)assign_map::is_assignable_data_type,
      (int)assign_map::is_assignable_dimension,
      (int)assign_map::is_assignable_layout,
      (int)assign_map::is_assignable_space
      );


    printf("%s\n",typeid(typename remote_view_t::device_type).name());
    printf("%s\n", typeid(typename traits::specialize).name());

    printf("%i, %s\n",Kokkos::Impl::ViewMapping<
      traits, typename remote_view_t::traits,
      typename traits::specialize>::is_assignable_data_type?1:0, typeid(typename traits::specialize).name());
}

int main(int argc, char *argv[]) {

  networking_init(argc, argv);

  Kokkos::initialize(argc, argv);
  {
    int myRank, nextRank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    nextRank = (myRank + 1) % numRanks;
    
    using traitUnmanaged_t = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using TeamPolicy_t = Kokkos::TeamPolicy<>;

    /*Regular host View type*/
    using regular_host_view_t = Kokkos::View<double *, Kokkos::HostSpace>;

    /*Regular View type*/
    using regular_view_t = Kokkos::View<double *>;

    /*Remote View type*/
    using remote_view_t = Kokkos::View<double*, RemoteSpace_t>;

    /*Kokkos Kernels View type*/
    using XVector_Internal_t = Kokkos::View<
      typename regular_view_t::non_const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<regular_view_t>::array_layout,
      typename regular_view_t::device_type, traitUnmanaged_t>;

    /*Kokkos Kernels View type*/
    using XVector_Internal_managed_t = Kokkos::View<
      typename regular_view_t::non_const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<regular_view_t>::array_layout,
      typename regular_view_t::device_type>;


    /*Kokkos Kernels View type derived from a remote_view_t */
    using XVector_Internal_Derived_t = Kokkos::View<
      typename remote_view_t::non_const_value_type *, 
      typename KokkosKernels::Impl::GetUnifiedLayout<remote_view_t>::array_layout,
      typename remote_view_t::device_type, traitUnmanaged_t>;
    
    
    /*Unmanaged remote View type */
    using umngt_remote_view_t = Kokkos::View<double*, RemoteSpace_t, traitUnmanaged_t>;

    /*Instances and example usage*/
    regular_view_t regular_view ("regular_view", SIZE);
    remote_view_t remote_view ("remote_view", SIZE);
    remote_view_t remote_view_2 = remote_view;
    remote_view_t remote_view_3;
    remote_view_3 = remote_view_2;
        
    //remote_view = regular_view; //Compile-time error
    //regular_view = remote_view; //Compile-time error

    XVector_Internal_t kernels_view(remote_view.data(), SIZE);
    XVector_Internal_managed_t kernels_view_managed(remote_view.data(), SIZE);
    XVector_Internal_Derived_t kernels_view_derived;

    using traits = typename XVector_Internal_Derived_t::traits;
    print_view_type_accessiblity(kernels_view_derived, remote_view);
        
    /*Use with subviews*/
    auto local_range = Kokkos::Experimental::get_range(SIZE, myRank);
    auto remote_range = Kokkos::Experimental::get_range(SIZE, nextRank); //get the range from the next rank
    
    auto local_sub_view = Kokkos::subview(remote_view, local_range);
    XVector_Internal_t kernels_view_valid(local_sub_view.data(), local_range.second - local_range.first);   

    auto remote_sub_view = Kokkos::subview(remote_view, remote_range);
    XVector_Internal_t kernels_view_invalid(local_sub_view.data(), remote_range.second - remote_range.first); //Runtime error (out of bounds)

    //Accessing remote data in bulk in device memory
    remote_view_t copied_data_from_remote("copied_data_from_remote", remote_range.second - remote_range.first);

    Kokkos::parallel_for(
    "Team", TeamPolicy_t(1, Kokkos::AUTO),
    KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
      Kokkos::single(Kokkos::PerThread(team), [&]() {
        Kokkos::Experimental::RemoteSpaces::local_deep_copy(team, copied_data_from_remote,
                                                            remote_sub_view);
      });
    });

    XVector_Internal_t kernels_view_with_data_from_remote (copied_data_from_remote.data(),remote_range.second - remote_range.first);

    //Accessing remote data in bulk on host memory
    //Init local range on device
    Kokkos::parallel_for(
    "Team", local_range.second - local_range.first,
    KOKKOS_LAMBDA(int i) {
      local_sub_view(i) = (double) myRank;
    });

    regular_host_view_t copied_data_from_remote_host("copied_data_from_remote_host",remote_range.second - remote_range.first);
    Kokkos::deep_copy(copied_data_from_remote_host, local_sub_view);
    //this should return correct data as we deep_copy a block of data from the local heap allocation

    if(myRank == 0)
      printf("A:%f (expected: %f)\n",copied_data_from_remote_host(SIZE/numRanks/2), (double) myRank);

    Kokkos::deep_copy(copied_data_from_remote_host, local_sub_view);
    //this should return wrong data as we deep_copy a block of data from a wrong ptr
    if(myRank == 0)
      printf("B:%f (expected: %f)\n",copied_data_from_remote_host(SIZE/numRanks/2), (double) nextRank);

    //Other tests
    #if 0 //Uncoment to see accessibility flags
    print_view_type_accessiblity(remote_view, regular_view);
    #endif

    /*Call into Kokkos Kernels*/
   // const double local_norm = KokkosBlas::nrm2_squared(kernels_view);
   // const double local_norm_2 = KokkosBlas::nrm2_squared(remote_view);
  
  }

  Kokkos::finalize();
  networking_fin();

}
