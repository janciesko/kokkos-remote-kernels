
void networking_init(int argc,char*argv[])
{
  int mpi_thread_level_required = MPI_THREAD_MULTIPLE;
  int mpi_thread_level_available;
  MPI_Init_thread(&argc, &argv, mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);

#ifdef KRS_ENABLE_SHMEMSPACE
  mpi_thread_level_required = SHMEM_THREAD_MULTIPLE;
  shmem_init_thread(mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
#endif

  MPI_Comm mpi_comm;

#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif
}

void networking_fin()
{
#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
#endif
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_finalize();
#else
  MPI_Finalize();
#endif
}
