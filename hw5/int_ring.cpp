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

  int Nrepeat = 10000; double ttavg = 0.0;
  for (int k = 0; k < Nrepeat; k++)
  {
    //Ring: Recieve value from mpirank-1, add yourself, send to mpirank+1 
    double tt = MPI_Wtime();
    int v = 0;
    if (mpirank != 0)
    { //Recieve and add yourself
      MPI_Recv(&v, 1, MPI_INT, mpirank - 1, mpirank, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      v += mpirank;
    }
    // Send your value to next guy in ring
    MPI_Send(&v, 1, MPI_INT, 
             (mpirank+1)%mpisize, (mpirank+1)%mpisize, 
             MPI_COMM_WORLD);
    if (mpirank == 0)
    { //Once we've come full circle, output answer.
      MPI_Recv(&v, 1, MPI_INT, mpisize - 1, mpirank, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      ttavg += MPI_Wtime() - tt;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (mpirank == 0)
  {
    printf("Latency: %f ms\n", 1000*ttavg / mpisize / Nrepeat );
  }
  //Exit workers, ring done.
  MPI_Finalize();
  return 0;
}
