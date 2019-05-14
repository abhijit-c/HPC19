// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Status status, status1;

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, p, processor_name);

  // Number of random numbers per processor 
  int N;
  sscanf(argv[1], "%d", &N);

  MPI_Barrier(MPI_COMM_WORLD);

  int* vec = (int*)malloc(N*sizeof(int));
  int *sdispls = (int *)calloc(sizeof(int), p);
  int *rdispls = (int *)malloc(p*sizeof(int));
  int *bkt_cnts = (int *)calloc(sizeof(int), p);
  int *inc_cnts = (int *)malloc(p*sizeof(int));
  int *sample = (int *)malloc( (p-1)*sizeof(int) );

  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  MPI_Barrier(MPI_COMM_WORLD);

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector

  for (int k = 0; k < p-1; ++k) { sample[k] = vec[(N/p) + p*k]; }

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int *world_samples = NULL;
  if (p == 0) 
  { 
    world_samples = (int *)malloc(sizeof(int)*(p-1)*p); 
  }
  MPI_Gather(&sample, p-1, MPI_INT, 
             &world_samples, p-1, MPI_INT, 0,
             MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if (p == 0) 
  { 
    std::sort(world_samples, world_samples + p*(p-1));
    for (int k = 0; k < p-1; ++k) 
    { 
      sample[k] = vec[(p-1)*(k+1)]; 
    } 
  } 

  // root process broadcasts splitters to all other processes
  MPI_Bcast(sample, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int count = 0;
  for (int k = 1; k < p; ++k)
  {
    sdispls[k] = std::lower_bound(vec, vec+N, sample[k-1])-vec;
    bkt_cnts[k-1] = sdispls[k] - count;
    count += sdispls[k];
  }
  bkt_cnts[p-1] = N - sdispls[p-1];

  MPI_Alltoall(&bkt_cnts[0], 1, MPI_INT,
               &inc_cnts[0], 1, MPI_INT, MPI_COMM_WORLD);

  count = inc_cnts[0];
  rdispls[0] = 0;
  for (int k = 1; k < p; k++)
  {
    rdispls[k] = inc_cnts[k-1] + rdispls[k-1];
    count += inc_cnts[k];
  }

  int *fvec = (int *)malloc(count*sizeof(int));
  MPI_Alltoallv(&vec[0], &bkt_cnts[0], &sdispls[0], MPI_INT,
                &fvec[0],&inc_cnts[0], &rdispls[0], MPI_INT, MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(fvec, fvec+count);

  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (rank==0) printf("Sample sort w/ N/p = %d: t = %.6f.\n", N, elapsed);

  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "ssort_output/%02d.out", rank);
  fd = fopen(filename,"w+");

  if(NULL == fd) {
    printf("Error opening file \n");
    return 1;
  }

  fprintf(fd, "Process %d has values :\n", rank);
  for(int i = 0; i < count; ++i) { fprintf(fd, "  %d\n", fvec[i]); }

  fclose(fd);

  free(vec);
  free(fvec);
  free(sample);
  free(sdispls);
  free(rdispls);
  free(bkt_cnts);
  free(inc_cnts);
  if (p == 0) { free(world_samples); }
  MPI_Finalize();
  return 0;
}
