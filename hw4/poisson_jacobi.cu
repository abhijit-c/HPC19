#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "utils.h"

#define BLOCK_SIZE 32

/* https://stackoverflow.com/a/14038590/5832371
 * CUDA GPU error checking.
*/
inline void gpuAssert(cudaError_t code, const char *file, int line, 
                      bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", 
            cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }


double presidual(const long N, const double *u, const double *f)
{ //Compute ||Au - f|| under the frobenius norm
    double ihsq = N*N;
    double resid = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:resid) 
    for (int j = 1; j < N-1; j++)
    {
        for (int i = 1; i < N-1; i++)
        {
            double v = ihsq * (- u[(i-1)+j*N] - u[i+(j-1)*N]
                               + 4*u[i+j*N]
                               - u[(i+1)+j*N] - u[i+(j+1)*N] )
                               - f[i+j*N];
            resid += v*v;
        }
    }
    return sqrt(resid);
}

/*
 * CPU Jacobi Poisson Step:
 * Computes one step of u[i,j] = (1/4)*(h^2 f[i,j] + u0[i-1,j]  + u0[i,j-1]  +
 * u0[i+1,j]  + u0[i,j+1]) for all 1 <= i, j < N-1.
 */
void jacobi_step_cpu(double *u, const double *u0, const double *f, const long N)
{
  double h = 1.0 / (double)N;
  #pragma omp parallel for collapse(2)
  for (long i = 1; i < N-1; i++)
  {
    for (long j = 1; j < N-1; j++)
    {
      u[i*N + j] = 0.25 * ( h*h*f[i*N + j] + u0[(i-1)*N + j] + 
                                             u0[i*N + (j-1)] + 
                                             u0[(i+1)*N + j] + 
                                             u0[i*N + (j+1)] ); 
    }
  }
}

/*
 * GPU Jacobi Poisson Step:
 * Computes one step of u[i,j] = (1/4)*(h^2 f[i,j] + u0[i-1,j]  + u0[i,j-1]  +
 * u0[i+1,j]  + u0[i,j+1]) for idx*BLOCK_SIZE <= i < (idx+1)*BLOCK_SIZE and
 * jdx*BLOCK_SIZE <= j < (jdx+1)*BLOCK_SIZE.
 */
__global__ void jacobi_step_gpu(
                double *u, const double *u0, const double *f, 
                const long N)
{
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x + 1;
  int jdx = (blockIdx.y) * blockDim.y + threadIdx.y + 1;
  double h = 1.0 / (double)N;
  u[idx*N + jdx] = 0.25 * ( h*h*f[idx*N + jdx] + u0[(idx-1)*N + jdx] + 
                                                 u0[idx*N + (jdx-1)] + 
                                                 u0[(idx+1)*N + jdx] + 
                                                 u0[idx*N + (jdx+1)] );
}
/*
 * Having learned safer GPU error handling practices, I try to be better here
 * code quality wise than my other two cuda functions. Also helps that I
 * understand this code MUCH more than whatever black magic that reduce method
 * is.
*/
int main(int argc, char** argv) 
{
  printf("Jacobi iteration with Cuda vs. CPU\n");

  const long N = 1<<10;
  const long N_grid = N+2; // Including ghost points
  const long MAX_ITERATESM1 = 1000;
  Timer t;

  // Malloc structures. Note we leave room for ghost points.
  double *f  = (double*) malloc(N_grid*N_grid*sizeof(double));
  double *u0 = (double*) malloc(N_grid*N_grid*sizeof(double));
  double *u  = (double*) malloc(N_grid*N_grid*sizeof(double));

  /* BEGIN CPU JACOBI POISSON */

  //Initialize vectors.  
  for (long i = 0; i < N_grid*N_grid; i++) { f[i] = 1; u0[i] = u[i] = 0; }

  printf("Initial residue: %.4e\n", presidual(N_grid, u, f));
  t.tic();

  for (int k = 0; k < MAX_ITERATESM1; k += 2)
  {
    jacobi_step_cpu(u, u0, f, N_grid);
    jacobi_step_cpu(u0, u, f, N_grid);
  }
  jacobi_step_cpu(u, u0, f, N_grid);

  double time = t.toc();
  printf("CPU computation: %.4f seconds: Final residue: %.4e\n", 
         time, presidual(N_grid, u, f) );

  // Reinitialize arrays
  for (long i = 0; i < N_grid*N_grid; i++) { f[i] = 1; u0[i] = u[i] = 0; }

  /* END CPU JACOBI POISSON */ /* BEGIN GPU JACOBI POISSON */

  // Allocate vectors onto GPU and transfer host data to device.
  double *f_d, *u_d, *u0_d;
  gpuErrchk( 
    cudaMalloc(&f_d, N_grid*N_grid*sizeof(double)) 
  );
  gpuErrchk( 
    cudaMemcpy(f_d, f, N_grid*N_grid*sizeof(double), cudaMemcpyHostToDevice) 
  );
  gpuErrchk( 
    cudaMalloc(&u_d, N_grid*N_grid*sizeof(double)) 
  );
  gpuErrchk( 
    cudaMemcpy(u_d, u, N_grid*N_grid*sizeof(double), cudaMemcpyHostToDevice) 
  );
  gpuErrchk( 
    cudaMalloc(&u0_d, N_grid*N_grid*sizeof(double)) 
  );
  gpuErrchk( 
    cudaMemcpy(u0_d, u0, N_grid*N_grid*sizeof(double), cudaMemcpyHostToDevice) 
  );
  cudaDeviceSynchronize();

  // Warp dimension calculation: code directly from cuda C programmers guide
  // Divide N into BLOCK_SIZE pieces, overshoot if not divisible.
  int GRID_SIZE = 0; // For some reason this breaks when BLOCK_SIZE defined.
  if ( N % BLOCK_SIZE == 0) { GRID_SIZE = N / BLOCK_SIZE; }
  else { GRID_SIZE = (N / BLOCK_SIZE) + 1; }

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(GRID_SIZE, GRID_SIZE); // From lecture9/filter.cu

  t.tic();

  for (long k = 0; k < MAX_ITERATESM1; k += 2)
  {
    jacobi_step_gpu<<<gridDim, blockDim>>>(u_d, u0_d, f_d, N_grid);
    jacobi_step_gpu<<<gridDim, blockDim>>>(u0_d, u_d, f_d, N_grid);
  }
  jacobi_step_gpu<<<gridDim, blockDim>>>(u_d, u0_d, f_d, N_grid);

  gpuErrchk( 
    cudaMemcpy(u, u_d, N_grid*N_grid*sizeof(double), cudaMemcpyDeviceToHost) 
  );
  cudaDeviceSynchronize();

  time = t.toc();
  printf("GPU computation: %.4f seconds: Final residue: %.4e\n", 
         time, presidual(N_grid, u, f) );

  /* END GPU JACOBI POISSON */

  free(f); free(u0); free(u);
  gpuErrchk( cudaFree(f_d) );
  gpuErrchk( cudaFree(u_d) );
  gpuErrchk( cudaFree(u0_d) );
}
