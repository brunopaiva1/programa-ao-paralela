/*16. [Pacheco and Malensek, 2022] Em um algoritmo de regra trapezoidal, se houver um total
de t threads, n trapézios e n for divisível por t, podemos atribuir n/t trapézios a cada
thread. Escreva um programa que implemente corretamente a regra trapezoidal quando
o número de trapézios for um múltiplo do número de threads.*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define THREADS_PER_BLOCK 256

__device__ __host__ double f(double x) {
    return cos(x) * x; 
}

__global__ void integrate(const double a, const double h, const long long n, const int threads_total, double *global_sum) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= threads_total) return;

    long long traps_per_thread = n / threads_total;
    double local_sum = 0.0;
    double start_x = a + thread_id * traps_per_thread * h;

    int start_index = (thread_id == 0) ? 1 : 0;

    for (long long i = start_index; i < traps_per_thread; i++) {
        double x = start_x + i * h;
        local_sum += f(x);
    }

    atomicAdd(global_sum, local_sum);
}

int main(int argc, char *argv[]) {
    clock_t start, end;
    start = clock();
    if (argc != 5) {
        printf("Uso: %s <threads> <trapezios> <a> <b>\n", argv[0]);
        return 1;
    }

    int threads_total = atoi(argv[1]);
    long long n = atoll(argv[2]);

    if (n % threads_total != 0) {
        printf("Erro: O numero de trapezios (%lld) deve ser divisivel pelo de threads (%d).\n", n, threads_total);
        return 1;
    }

    double a = atof(argv[3]);
    double b = atof(argv[4]);
    double h = (b - a) / n;

    double integral_result = 0.5 * (f(a) + f(b));
    
    int blocks = (threads_total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    double *d_global_sum;
    cudaMalloc(&d_global_sum, sizeof(double));
    cudaMemcpy(d_global_sum, &integral_result, sizeof(double), cudaMemcpyHostToDevice);

    integrate<<<blocks, THREADS_PER_BLOCK>>>(a, h, n, threads_total, d_global_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(&integral_result, d_global_sum, sizeof(double), cudaMemcpyDeviceToHost);
    
    integral_result *= h;

    printf("Resultado da integral: %.12f\n", integral_result);

    cudaFree(d_global_sum);

    end = clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tempo total (CPU + GPU): %f segundos\n", total_time);

    return 0;
}