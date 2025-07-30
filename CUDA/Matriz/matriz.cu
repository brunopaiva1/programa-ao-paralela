/*18. Implemente uma soma de duas matrizes que utilize todas GPUs disponíveis em nó. Este
exercício deve ser executado em uma máquina com ao menos duas GPUs. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

__global__ void add_matrices(const float *A, const float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamanho_N>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Valor de N inválido.\n");
        return 1;
    }

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        printf("Nenhuma GPU disponível.\n");
        return 1;
    }

    printf("Usando %d GPUs.\n", num_gpus);

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    
    omp_set_num_threads(num_gpus);

    #pragma omp parallel
    {
        int i = omp_get_thread_num();

        cudaSetDevice(i);

        int rows_per_gpu = N / num_gpus;
        int start_row = i * rows_per_gpu;
        
        if (i == num_gpus - 1) {
            rows_per_gpu = N - start_row;
        }
        
        int chunk_size = rows_per_gpu * N;
        size_t bytes = chunk_size * sizeof(float);

        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, bytes);
        cudaMalloc((void **)&d_B, bytes);
        cudaMalloc((void **)&d_C, bytes);

        cudaMemcpy(d_A, A + start_row * N, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B + start_row * N, bytes, cudaMemcpyHostToDevice);

        dim3 dimBlock(16, 16);
        dim3 dimGrid((N + 15) / 16, (rows_per_gpu + 15) / 16);

        add_matrices<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(C + start_row * N, d_C, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    printf("C[0] = %f\n", C[0]);
    printf("C[N*N-1] = %f\n", C[N * N - 1]);

    free(A);
    free(B);
    free(C);

    return 0;
}