#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) 
{
  //MPI Setup
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  int Nrepeat = 1000; int numel = (1 << 19); double ttavg = 0.0;
  for (int k = 0; k < Nrepeat; k++)
  {
    //Ring: Recieve value from mpirank-1, add yourself, send to mpirank+1 
    int *arr = (int *)malloc( numel * sizeof(int) ); 
    for (int i = 0; i < numel; i++) { arr[i] = 0; }
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();
    if (mpirank != 0)
    { //Recieve and add yourself
      MPI_Recv(arr, numel, MPI_INT, mpirank - 1, mpirank, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      for (int i = 0; i < numel; i++) { arr[i] += mpirank; }
    }
    // Send your value to next guy in ring
    MPI_Send(arr, numel, MPI_INT, 
             (mpirank+1)%mpisize, (mpirank+1)%mpisize, 
             MPI_COMM_WORLD);
    if (mpirank == 0)
    { //Once we've come full circle, output answer.
      MPI_Recv(arr, numel, MPI_INT, mpisize - 1, mpirank, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      ttavg += MPI_Wtime() - tt;
    }
    free(arr);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (mpirank == 0)
  {
    printf("Bandwidth: %f Gbytes/s\n", 
           Nrepeat*mpisize*sizeof(int)*numel / ttavg / 1e9);
  }
  //Exit workers, ring done.
  MPI_Finalize();
  return 0;
}
