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
  MPI_Request request;

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
  /*
  MPI_Isend(&v, 1, MPI_INT, 
           (mpirank+1)%mpisize, (mpirank+1)%mpisize, 
           MPI_COMM_WORLD, &request);
  */
  MPI_Send(&v, 1, MPI_INT, 
           (mpirank+1)%mpisize, (mpirank+1)%mpisize, 
           MPI_COMM_WORLD);

  if (mpirank == 0)
  { //Once we've come full circle, output answer.
    MPI_Recv(&v, 1, MPI_INT, mpisize - 1, mpirank, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    tt = MPI_Wtime() - tt;
    printf("Recieved %d, Truth = %d. Compute time = %f.\n",
           v, mpisize*(mpisize-1)/2, tt );
    printf("Latency: %e ms\n",
           1000*tt / mpisize );
  }

  //Exit workers, algorithm done.
  MPI_Finalize();
  return 0;
}
