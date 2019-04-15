#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <math.h>

double vector_norm(double *a, long N)
{
  double norm = 0.0;
  #pragma omp parallel for schedule(static) reduction(+:norm)
  for (long k = 0; k < N; k++) 
  { 
    double t = a[k];
    norm += t*t; 
  }
  return sqrt(norm);
}

double vector_err(double *a, double *b, long N)
{
  double norm = 0.0;
  #pragma omp parallel for schedule(static) reduction(+:norm)
  for (long k = 0; k < N; k++) 
  { 
    double t = a[k] - b[k];
    norm += t*t; 
  }
  return sqrt(norm);
}

void mvec_cpu(double *y, const double* A, const double* x, long N)
{ //Compute y = A*x where A is (n X n) and x is (n X 1) row major.
  for (long k = 0; k < N; k++)
  { // y_k = sum_i A_ki * x_i = dot(A_k:, x)
    double sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (long i = 0; i < N; i++) { sum += A[i+k*N]*x[i]; }
    y[k] = sum;
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

__global__ void reduction_kernel(double* sum, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx]*b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}
__global__ void reduction_kernel(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

int main() {
  long N = (1UL<<5);

  // Allocate and initialize matrices and vectors.
  double *A, *x, *y_c, *y_g;
  cudaMallocHost((void**)&A, N * N * sizeof(double)); // A row-major matrix
  cudaMallocHost((void**)&x, N * sizeof(double)); // x vector
  cudaMallocHost((void**)&y_c, N * sizeof(double)); // cpu resultant vector
  cudaMallocHost((void**)&y_g, N * sizeof(double)); // gpu resultant vector
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) { x[i] = y_c[i] = y_g[i] = 1.0/sqrt(N); }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) { A[i] = 1.0; }

  // Compute CPU Matrix Vector Product.
  //double dot_ref, dot;
  double tt = omp_get_wtime();
  mvec_cpu(y_c, A, x, N);
  printf("CPU Bandwidth = %f GB/s\n", 1*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  // Initialize and allocate GPU matrices and vectors
  double *x_d, *A_d, *temp_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&A_d, N*N*sizeof(double));
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&temp_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks

  // Copy host matrices to GPU
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Begin Matrix Vector Product
  tt = omp_get_wtime();

  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  for (long k = 0; k < N; k++)
  {
    double* sum_d = temp_d;
    reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, A_d+N*k, x_d, N);
    while (Nb > 1) 
    {
      long M = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d + M, sum_d, N);
      sum_d += M;
    }
    cudaMemcpyAsync(&(y_g[k]), sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  }

  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 1*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  for (long k = 0; k < 10; k++)
  {
    printf("k=%ld\t y_g[k] = %f\n", k, y_g[k]);
  }

  printf("||y_c|| = %f\n", vector_norm(y_c, N));
  printf("||y_g|| = %f\n", vector_norm(y_g, N));
  printf("Error = %f\n", vector_err(y_c, y_g, N));


  cudaFree(x_d);
  cudaFree(A_d);
  cudaFree(temp_d);
  cudaFreeHost(x);
  cudaFreeHost(A);
  cudaFreeHost(y_c);
  cudaFreeHost(y_g);

  return 0;
}
