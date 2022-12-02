
void networking_init(int argc,char*argv[])
{
  MPI_Init(&argc, &argv);
#ifdef KRS_ENABLE_SHMEMSPACE
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
