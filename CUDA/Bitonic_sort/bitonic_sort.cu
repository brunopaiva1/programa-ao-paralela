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


