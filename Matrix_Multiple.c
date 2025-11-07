#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef N
#define N 2048
#endif

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

static void matmul_seq(void) {
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            double aik = A[i][k];
            for (int j = 0; j < N; ++j)
                C[i][j] += aik * B[k][j];
        }
}

int main(void) {
    matrixInit();
    clock_t t1 = clock();
    matmul_seq();
    clock_t t2 = clock();
    printf("ST clock ticks: %ld\n", (long)(t2 - t1));
    return 0;
}
