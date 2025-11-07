#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef N
#define N 2048
#endif
#ifndef BS
#define BS 64
#endif

#define MIN(a,b) ((a)<(b)?(a):(b))
#define FACTOR 1.1
static double A[N][N], B[N][N], C[N][N];

static void matrixInit(void) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            srand(i + j);
            A[i][j] = (rand() % 10) * FACTOR;
            B[i][j] = (rand() % 10) * FACTOR;
            C[i][j] = 0.0;
        }
}

static void matmul_block_omp(void) {
    // Parallelize across tiles (ii, jj)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += BS)
        for (int jj = 0; jj < N; jj += BS) {
            for (int kk = 0; kk < N; kk += BS) {
                int iMax = MIN(ii + BS, N);
                int jMax = MIN(jj + BS, N);
                int kMax = MIN(kk + BS, N);
                for (int i = ii; i < iMax; ++i)
                    for (int k = kk; k < kMax; ++k) {
                        double aik = A[i][k];
                        for (int j = jj; j < jMax; ++j)
                            C[i][j] += aik * B[k][j];
                    }
            }
        }
}

int main(void) {
    matrixInit();
    double t1 = omp_get_wtime();
    matmul_block_omp();
    double t2 = omp_get_wtime();
    printf("OMP-b seconds: %.6f  (threads=%d, N=%d, BS=%d)\n",
           t2 - t1, omp_get_max_threads(), N, BS);
    return 0;
}
