// File Name: 1d_laplace.cc
// Author: Abhijit Chowdhary (ac6361)

#include <stdio.h>
#include <math.h>
#include "utils.h"

#define NUM_ITER 5000

double ResidLaplace(long N, double *u, double *f)
{ //Compute ||Au - f|| under the Euclidean Norm
    double hsq = 1.0/((N+1)*(N+1)), a_diag = (2.0/(hsq)), a_bdiag = -1.0/hsq;
    double v = 0.0, tmp = 0.0;

    tmp = u[0]*a_diag + u[1]*a_bdiag - f[0]; v += tmp*tmp;
    tmp = ( u[N-1]*a_diag + u[N-2]*a_bdiag ) - f[N-1]; v += tmp*tmp;
    for (int i = 1; i < N-1; i++)
    {
        tmp = ( a_diag*u[i] + a_bdiag*(u[i-1] + u[i+1]) ) - f[i]; 
        v += tmp*tmp;
    }
    return sqrt(v);
}

/* Numerically approximates the solution to:
 * -u'' = f in (0,1) with u(0) = 0 and u(1) = 0
 * using the Jacobi method.
 * INPUT: long N: Number of discretization points
 *        double tol: Tolerance to compute solution to.
 *        double *f: discretization of f with N points
 *        double *u0: Initial guess vector for u
 * OUTPUT: double *u: Array of size N holding solution to equation.
 *         long it: Number of iterations needed to hit tol.
 */
long Laplace1DJacobi(long N, double tol, double *f, double *u0, double *u)
{ //Proceed via Jacobi iteration: 
    double hsq = 1.0/((N+1)*(N+1)), a_diag = (2.0/(hsq)), a_bdiag = -1.0/hsq;
    double nu[N];
    for (long k = 0; k < N; k++) { u[k] = u0[k]; nu[k] = u0[k]; }

    //Compute u_i^{k+1} = 1/a_ii (f_i - sum_{j \neq i} a_ij u_j^k
    long it = 0;
    double cur_res = 0.0;
    while (ResidLaplace(N, u, f) > tol && it < 100)
    { //Note for laplace, the summation only has 2 terms max.
        nu[0] = (1/a_diag)*(f[0] - a_bdiag*u[1]);
        nu[N-1] = (1/a_diag)*(f[N-1] - a_bdiag*u[N-2]);
        for (long i = 1; i < N-1; i++)
        {
            nu[i] = (1/a_diag)*( f[i] - a_bdiag*(u[i-1] + u[i+1]) );
        }
        for (long j = 0; j < N; j++) { u[j] = nu[j]; }
        cur_res = ResidLaplace(N,u,f);
        it = it + 1;
        if (it % 10000 == 0) 
        { 
            printf("At iteration %ld err = %10f\n",it, cur_res);
        }
    }
    return it;
}

/* Numerically approximates the solution to:
 * -u'' = f in (0,1) with u(0) = 0 and u(1) = 0
 * using the Gauss Siedel method.
 * INPUT: long N: Number of discretization points
 *        double tol: Tolerance to compute solution to.
 *        double *f: discretization of f with N points
 *        double *u0: Initial guess vector for u
 * OUTPUT: double *u: Array of size N holding solution to equation.
 *         long it: Number of iterations needed to hit tol.
 */
long Laplace1DGaussSeidel(long N, double tol, double *f, double *u0, double *u)
{ //Proceed via Seidel iteration: 
    double hsq = 1.0/((N+1)*(N+1)), a_diag = (2.0/(hsq)), a_bdiag = -1.0/hsq;
    double nu[N];
    for (long k = 0; k < N; k++) { u[k] = u0[k]; nu[k] = u0[k]; }

    long it = 0;
    double cur_res = 0.0;
    while (ResidLaplace(N, u, f) > tol && it < 100)
    { //Note for laplace, the summation only has 2 terms max.
        nu[0] = (1/a_diag)*(f[0] - a_bdiag*u[1]);
        nu[N-1] = (1/a_diag)*(f[N-1] - a_bdiag*nu[N-2]);
        for (long i = 1; i < N-1; i++)
        {
            nu[i] = (1/a_diag)*( f[i] - a_bdiag*(nu[i-1] + u[i+1]) );
        }
        for (long j = 0; j < N; j++) { u[j] = nu[j]; }
        cur_res = ResidLaplace(N,u,f);
        it = it + 1;
        if (it % 10000 == 0) 
        { 
            printf("At iteration %ld err = %10f\n",it, cur_res);
        }
    }
    return it;
}

int main(int argc, char** argv) 
{
    long N = read_option<long>("-n", argc, argv);
    double *f = (double*) malloc(N*sizeof(double));
    double *u0 = (double*) malloc(N*sizeof(double));
    double *u = (double*) malloc(N*sizeof(double));

    
    //Initialize vectors.
    for (long i = 0; i < N; i++)
    {
        f[i] = 1; u0[i] = 0; u[0] = 0;
    }
    long it;
    double ores = ResidLaplace(N,u,f);
    Timer t;
    printf("Original Error: %10f\n", ores);
    t.tic();
    it = Laplace1DJacobi(N, ores / 1e6, f, u0, u);
    double time = t.toc();
    printf(
        "Jacobi Laplace Solve for N = %ld and %ld iterations: %10fs.\n",
        N, it, time);
    printf("Error: %10f\n", ResidLaplace(N, u, f));

    //Initialize vectors.
    for (long i = 0; i < N; i++)
    {
        f[i] = 1; u0[i] = 0; u[0] = 0;
    }
    it;
    ores = ResidLaplace(N,u,f);
    printf("Original Error: %10f\n", ores);
    t.tic();
    it = Laplace1DGaussSeidel(N, .5, f, u0, u);
    time = t.toc();
    printf(
        "Gauss-Seidel Laplace Solve for N = %ld and %ld iterations: %10fs.\n",
        N, it, time);
    printf("Error: %10f\n", ResidLaplace(N, u, f));
    free(f); free(u0); free(u);
}
