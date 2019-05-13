/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

int log4(int n)
{
  int cnt = 0;
  while (n >>= 2) { cnt++; }
  return cnt;
}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i, j;
  double tmp, gres = 0.0, lres = 0.0;

  for (j = 1; j <= lN; j++)
  {
      for (i = 1; i <= lN; i++)
      {
          tmp = invhsq * (- lu[(i-1)+j*(lN+2)] - lu[i+(j-1)*(lN+2)]
                             + 4*lu[i+j*(lN+2)]
                             - lu[(i+1)+j*(lN+2)] - lu[i+(j+1)*(lN+2)] )
                             - 1;
          lres += tmp * tmp;
      }
  }
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}

int main(int argc, char * argv[])
{
  int mpirank, i, p, N, lN, N_l, iter, max_iters;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N_l);
  sscanf(argv[2], "%d", &max_iters);

  int J = log4(p);
  int j2 = (1<<J);

  N = (2<<J)*N_l;

  /* compute number of unknowns handled by each process */
  lN = N_l*N_l;

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN+2)*(lN+2));
  double * lunew = (double *) calloc(sizeof(double), (lN+2)*(lN+2));
  double * lutemp;

  double h = 1.0 / ( (N + 1)*(N + 1) );
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) 
  {

    /* Jacobi step for local points */
    for (int j = 1; j <= lN; j++)
    {
      for (i = 1; i <= lN; i++)
      {
        lunew[i+j*(N+2)] = 0.25*(hsq + 
                                lunew[(i-1)+j*(N+2)] + lunew[i+(j-1)*(N+2)] + 
                                lunew[(i+1)+j*(N+2)] + lunew[i+(j+1)*(N+2)]); 
      }
    }

    /* communicate ghost values */
    if (mpirank < p - j2) 
    {
      /* If not the top processes, send/recv bdry values upward */
      MPI_Send(&(lunew[(lN+2)*lN]), lN+2, 
               MPI_DOUBLE, mpirank+j2, 124, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[(lN+2)*(lN+1)]), lN+2, 
               MPI_DOUBLE, mpirank+j2, 123, MPI_COMM_WORLD, &status);
    }
    if (mpirank >= j2) 
    {
      /* If not the bottom processes, send/recv bdry values downward */
      MPI_Send(&(lunew[lN+2]), lN+2, 
               MPI_DOUBLE, mpirank-j2, 123, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[0]), lN+2, 
               MPI_DOUBLE, mpirank-j2, 124, MPI_COMM_WORLD, &status);
    }
    double ghstore[lN+2];
    if (mpirank % j2 != 0) 
    {
      /* If not the left processes, send/recv bdry values leftward */
      for (int k = 0; k < lN+2 ; k++) { ghstore[k] = lunew[1+k*(lN+2)]; }
      MPI_Send(&(ghstore[0]), lN+2, 
               MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD);
      MPI_Recv(&(ghstore[0]), lN+2, 
               MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD, &status);
      for (int k = 0; k < lN+2 ; k++) { lunew[k] = ghstore[k]; } 
    }
    if (mpirank % j2 != j2 - 1) 
    {
      /* If not the right processes, send/recv bdry values rightward */
      for (int k = 0; k < lN+2 ; k++) { ghstore[k] = lunew[lN+k*(lN+2)]; }
      MPI_Send(&(ghstore[0]), lN+2, 
               MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD);
      MPI_Recv(&(ghstore[0]), lN+2, 
               MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD, &status);
      for (int k = 0; k < lN+2 ; k++) { lunew[(lN+1)+k*(lN+2)] = ghstore[k]; } 
    }

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) 
    {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) 
      {
        printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
