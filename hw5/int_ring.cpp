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

  //Ring: Recieve value from mpirank-1, add yourself, send to mpirank+1 
  int v;
  if (mpirank == 0)
  {
    v = 0;
  }
  else
  {
    MPI_Recv(&v, 1, MPI_INT, mpirank - 1, mpirank, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    v += mpirank;
  }
  MPI_Send(&v, 1, MPI_INT, 
           (mpirank+1)%mpisize, (mpirank+1)%mpisize, 
           MPI_COMM_WORLD);

  if (mpirank == 0)
  {
    MPI_Recv(&v, 1, MPI_INT, mpisize - 1, mpirank, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    printf("Zero recieved %d, and it should have been %d\n",
           v, mpisize*(mpisize-1)/2);
  }

  MPI_Finalize();
  return 0;
}
