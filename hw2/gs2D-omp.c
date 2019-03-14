// File Name: jacobi2D-omp.c
// Author: Abhijit Chowdhary (ac6361)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#ifndef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
#endif

#define NUM_ITER 5000
#define TOL 0.01

void print_mat(int N, int M, double *vec)
{
    printf("[\n");
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++)
        {
            printf("%.4f, ", vec[i+j*M]);
        }
        printf("\n");
    }
    printf("]\n");
}

void copy_into(int N, double *u1, double *u2)
{
    for (int k = 0; k <= N; k++) { u2[k] = u1[k]; }
}

double lresidual(int N, double *u, double *f)
{ //Compute ||Au - f|| under the frobenius norm
    double ihsq = (N+1)*(N+1);
    double resid = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:resid) 
    for (int j = 1; j <= N; j++)
    {
        for (int i = 1; i <= N; i++)
        {
            double v = ihsq * (- u[(i-1)+j*(N+2)] - u[i+(j-1)*(N+2)]
                               + 4*u[i+j*(N+2)]
                               - u[(i+1)+j*(N+2)] - u[i+(j+1)*(N+2)] )
                               - f[i+j*(N+2)];
            resid += v*v;
        }
    }
    return sqrt(resid);
}

int GaussSeidel(int N, double *u0, double *f, double *u)
{
    double init_res = lresidual(N, u0, f), res = init_res;
    double h = 1.0/(N+1);
    int it = 0;

    double *utmp = (double *)malloc((N+2)*(N+2)* sizeof(double));
    copy_into( (N+2)*(N+2), u0, u );

    while (res > init_res*TOL && it++ < NUM_ITER + 1)
    { //Run until desired tolerance, or until too long.
        #pragma omp parallel for
        for (int j = 1; j <= N; j++)
        { // Red point update
            for (int i = 2 - (j&1); i <= N; i+=2)
            {
                utmp[i+j*(N+2)] = 0.25*(h*h*f[ i + j*(N+2) ] + 
                                        utmp[ (i-1) + j*(N+2) ] + 
                                        utmp[ i + (j-1)*(N+2) ] +
                                        utmp[ (i+1) + j*(N+2) ] +
                                        utmp[ i + (j+1)*(N+2) ]);
            }
        }
        #pragma omp parallel for
        for (int j = 1; j <= N; j++)
        { // Black point update
            for (int i = 1 + (j&1); i <= N; i+=2)
            {
                utmp[i+j*(N+2)] = 0.25*(h*h*f[ i + j*(N+2) ] + 
                                        utmp[ (i-1) + j*(N+2) ] + 
                                        utmp[ i + (j-1)*(N+2) ] +
                                        utmp[ (i+1) + j*(N+2) ] +
                                        utmp[ i + (j+1)*(N+2) ]);
            }
        }
        copy_into( (N+2)*(N+2), utmp, u );
    }
    free(utmp);
    return it;
}

int main(int argc, char** argv) 
{
    printf("Gauss-Seidel iteration with parallelism\n");
    int N = read_option<long>("-n", argc, argv);
    int nghost = (N+2)*(N+2);
    Timer t;

    // Malloc structures. Note we leave room for ghost points.
    double *f = (double*) malloc(nghost*sizeof(double));
    double *u0 = (double*) malloc(nghost*sizeof(double));
    double *u = (double*) malloc(nghost*sizeof(double));

    //Initialize vectors.
    for (long i = 0; i < nghost; i++)
    {
        f[i] = 1; u0[i] = 0; u[i] = 0;
    }

    printf("Initial residue: %.4e\n", lresidual(N, u, f));
    t.tic();
    int its = GaussSeidel(N, u0, f, u);
    double time = t.toc();
    printf("%d iterations and %.4f seconds: Final residue: %.4e\n", its, time, lresidual(N, u, f));

    free(f); free(u0); free(u);
}
