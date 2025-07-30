#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <cuda_runtime.h>

int check_order(int *array, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (array[i] > array[i + 1]) {
            return 0;
        }
    }
    return 1;
}

int pot_2(int n) {
    int pot = 1;
    while (pot < n) {
        pot *= 2;
    }
    return pot;
}

int *generate_random_array(int n, int size) {
    int *array = (int *)malloc(size * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        array[i] = rand() % 20000;
    }
    for(int i = n; i < size; i++) {
        array[i] = INT_MAX;
    }
    return array;
}

void imprimi_array(int *array, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

__global__ void bitonic_sort(int *array, int stage, int bf_size, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    int id_partiner = i ^ stage;

    if(id_partiner > i){
        if(((i & bf_size) == 0 && array[i] > array[id_partiner]) ||
           ((i & bf_size) != 0 && array[i] < array[id_partiner])){
            int temp = array[i];
            array[i] = array[id_partiner];
            array[id_partiner] = temp;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Invalid size\n");
        return EXIT_FAILURE;
    }
    srand(time(NULL));

    int size = pot_2(n);
    int *h_array = generate_random_array(n, size);
    int *d_array;

    int print_size = (n < 32) ? n : 32;
    printf("Array input (first %d elements of %d):\n", print_size, size);
    imprimi_array(h_array, print_size);

    size_t size_bytes = size * sizeof(int);
    cudaMalloc((void **)&d_array, size_bytes);
    cudaMemcpy(d_array, h_array, size_bytes, cudaMemcpyHostToDevice);

    const int THREADS_PER_BLOCK = 256;
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int bf_size = 2; bf_size <= size; bf_size *= 2) {
        for (int stage = bf_size / 2; stage > 0; stage /= 2) {
            bitonic_sort<<<blocks, THREADS_PER_BLOCK>>>(d_array, stage, bf_size, size);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_array, d_array, size_bytes, cudaMemcpyDeviceToHost);

    printf("\nArray output (first %d elements):\n", print_size);
    imprimi_array(h_array, print_size);

    if (check_order(h_array, n)) {
        printf("\nThe array is sorted correctly.\n");
    } else {
        printf("\nThe array is NOT sorted correctly.\n");
    }

    free(h_array);
    cudaFree(d_array);

    return 0;
}