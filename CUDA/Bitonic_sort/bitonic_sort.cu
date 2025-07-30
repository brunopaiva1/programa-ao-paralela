/* 17. [Pacheco and Malensek, 2022] Implemente uma ordenação bitônica na qual cada thread
é responsável por dois blocos de elementos. Se o array tiver n elementos e houver
blk_ct blocos de threads e th_per_blk threads por bloco, considere que o número total
de threads é uma potência de dois e que n é divisível pelo número de threads. Assim
chunk_sz = n
blk_ct ×th_per_blk (2)
é um inteiro.
Cada thread é responsável por uma sublista contígua de chunk_sz elementos, e cada
thread inicialmente classificará sua sublista em ordem crescente. Então, se as threads t
e u forem pareadas para uma divisão e mesclagem, t < u, e t e u estiverem trabalhando
em uma sequência crescente, elas mesclarão suas sublistas em uma sequência cres-
cente, com t mantendo a metade inferior e u mantendo a metade superior. Se estiverem
trabalhando em uma sequência decrescente, t manterá a metade superior e u manterá a
metade inferior. Portanto, após cada divisão e mesclagem, cada thread sempre terá uma
sublista crescente.
Primeiro implemente a ordenação bitônica usando um único bloco de threads. Em se-
guida, modifique o programa para que ele possa lidar com um número arbitrário de blocos
de threads */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

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
        array[i] = rand() % (INT_MAX/10000000);
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

__global__ void bitonic_sort(int *array, int stage, int bf_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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

    printf("Array input (first %d elements): \n", n);
    imprimi_array(h_array, n);

    cudaMalloc((void **)&d_array, size * sizeof(int));
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int bf_size = 2; bf_size <= size; bf_size *= 2) {
        for (int stage = bf_size / 2; stage > 0; stage /= 2) {
            bitonic_sort<<<blocks, THREADS_PER_BLOCK>>>(d_array, stage, bf_size);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nArray output (first %d elements):\n", n);
    imprimi_array(h_array, n);
    
    if (check_order(h_array, n)) {
        printf("\n The array is sorted correctly.\n");
    } else {
        printf("\n The array is NOT sorted correctly.\n");
    }

    free(h_array);
    cudaFree(d_array);

    return EXIT_SUCCESS;
}

