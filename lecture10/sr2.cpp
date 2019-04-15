/* Send and receive example
 * Exchanging send and receive may
 * lead to deadlock
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int message_out = 42;
    int message_in;
    MPI_Status status;

    MPI_Recv(&message_in,  1, MPI_INT, 2, 999, MPI_COMM_WORLD, &status);
    MPI_Send(&message_out, 1, MPI_INT, 2, 999, MPI_COMM_WORLD);

    printf("Rank %d received %d\n", rank, message_in);
  } else if (rank == 2) {
    int message_out = 80;
    int message_in;
    MPI_Status status_2;

    MPI_Send(&message_out, 1, MPI_INT, 0, 999, MPI_COMM_WORLD);
    MPI_Recv(&message_in,  1, MPI_INT, 0, 999, MPI_COMM_WORLD, &status_2);

    printf("Rank %d received %d\n", rank, message_in);
  }

  MPI_Finalize();

  return 0;
}
