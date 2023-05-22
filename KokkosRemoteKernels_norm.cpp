#include "Kokkos_Core.hpp"
#include "KokkosBlas1_nrm2_squared.hpp"
#include "Kokkos_RemoteSpaces.hpp"

#include <mpi.h>
#include <helper.hpp>

//#define MY_LOCAL_DEBUG

using RemoteSpace_t  = Kokkos::Experimental::DefaultRemoteMemorySpace;
#define SIZE 1024

template <class DstTraits, class SrcTraits>
void print_view_type_accessiblity(DstTraits dV, SrcTraits sV)
{
    using dst_traits = typename DstTraits::traits;
    using src_traits = typename SrcTraits::traits;

    using assign_map = typename Kokkos::Impl::ViewMapping<
          dst_traits, src_traits, Kokkos::Experimental::RemoteSpaceSpecializeTag>;

    printf("%i %i\n", 
      (int)assign_map::is_assignable,
      (int)assign_map::is_assignable_data_type
      );
}

int main(int argc, char *argv[]) {

  networking_init(argc, argv);

  Kokkos::initialize(argc, argv);
  {
    
    using traitUnmanaged_t = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    
    /* Regular host View type */
    using regular_host_view_t = Kokkos::View<double *, Kokkos::HostSpace>;

    /* Regular View type */
    using regular_view_t = Kokkos::View<double *>;

    /* Remote View type */
    using remote_view_t = Kokkos::View<double*, RemoteSpace_t>;

    /* Compatible Remote View type */
    using remote_view_compat_t = Kokkos::View<double*, typename KokkosKernels::Impl::GetUnifiedLayout<remote_view_t>::array_layout, typename remote_view_t::device_type>;

    /* Kokkos Kernels View type derived from a remote_view_t
       This is what actually happens in Kokkos Kernels */
    using XVector_Internal_Derived_t = Kokkos::View<
      typename remote_view_t::non_const_value_type *,  
      typename KokkosKernels::Impl::GetUnifiedLayout<remote_view_t>::array_layout,
      typename remote_view_t::device_type, traitUnmanaged_t>;
  
    
    /* Create View */
    remote_view_compat_t remote_view ("remote_view", SIZE);

    /* Test Assignements */
    XVector_Internal_Derived_t xvector_derived_1 (remote_view);
    XVector_Internal_Derived_t xvector_derived_2  = remote_view;

    #ifdef MY_LOCAL_DEBUG
    print_view_type_accessiblity(xvector_orig, remote_view);
    #endif

    /* Call into Kokkos Kernels */
    const double local_norm = KokkosBlas::nrm2_squared(remote_view);  
  }

  Kokkos::finalize();
  networking_fin();
}

#undef SIZE

#ifdef MY_LOCAL_DEBUG
#undef MY_LOCAL_DEBUG
#endif