/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
    int nthreads, tid;
    float total;
    /*** Spawn parallel region ***/
    #pragma omp parallel private(tid)
    {
        /* Obtain thread number */
        tid = omp_get_thread_num();
        /* Only master thread does this */
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        printf("Thread %d is starting...\n",tid);

        total = 0.0;
        #pragma omp barrier

        /* do some work */
        #pragma omp for schedule(dynamic,10) reduction(+:total)
        for (int i=0; i<1000000; i++) { 
            total = total + i*1.0;
        }
        printf ("Thread %d is done! Total= %e\n",tid,total);
    } /*** End of parallel region ***/
}

/* There were a few problems here, mostly race conditions. 
 * First, to avoid the condition on the for loop index i, we moved it's
 * initialization inside the for loop. Equivalently, we could have marked it
 * private.
 * Second, we needed to mark total as an accumulation variable, therefore we
 * added a reduction(+:total) statement.
 * Finally, we moved the total = 0.0; statement before the barrier, so that the
 * threads wouldn't accidently reset total, since it's a shared variable.
 */
