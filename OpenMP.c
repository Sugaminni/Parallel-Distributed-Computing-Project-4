#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef N
#define N 2048
#endif

#define FACTOR 1.1
static double A[N][N], B[N][N], C[N][N];

// Keep init single-threaded so results match the sequential version exactly.
static void matrixInit(void) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            srand(i + j);
            A[i][j] = (rand() % 10) * FACTOR;
            B[i][j] = (rand() % 10) * FACTOR;
            C[i][j] = 0.0;
        }
}

// Parallelize (i,j); each thread computes distinct C[i][j]
static void matmul_omp(void) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

int main(void) {
    matrixInit();
    double t1 = omp_get_wtime();
    matmul_omp();
    double t2 = omp_get_wtime();
    printf("OMP seconds: %.6f  (threads=%d, N=%d)\n",
           t2 - t1, omp_get_max_threads(), N);
    return 0;
}
