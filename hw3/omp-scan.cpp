#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// I wanted a thread count that worked outside of the parallel region, so I
// found this hack on stack exchange:
// https://stackoverflow.com/a/13328691/5832371
int _omp_thread_count()
{
  int n = 0;
  #pragma omp parallel reduction(+:n)
    n += 1;
  return n;
}

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  int num_threads = _omp_thread_count();
  if (n < num_threads) { scan_seq(prefix_sum, A, n); }
  
  #pragma omp parallel
  { //Obj: Scan region [l0, ln]
    int tid = omp_get_thread_num();
    long l0 = ( tid )*(n/num_threads); 
    long ln = (tid+1==num_threads) ? n : (tid+1)*(n/num_threads); 
    printf("[%d, %d]\n", l0, ln);
    scan_seq(prefix_sum+l0, A+l0, ln-l0);
  }

  
  for (int tid = 1; tid < num_threads; tid++)
  { //Correct regiod of tid
    long l0 = ( tid )*(n/num_threads); 
    long ln = (tid+1==num_threads) ? n : (tid+1)*(n/num_threads); 
    for (long k = l0; k < ln; k++) 
    { 
      prefix_sum[k] += prefix_sum[l0-1]; 
    }
  }
  
}

int main() {
  //long N = 1000000;
  long N = 16;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) 
  {
    err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("%d vs %d\n", B0[i], B1[i]);
  }
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
