#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

int pot_2(int n) {
    int i = 1;
    while (i < n) {
        i *= 2;
    }
    return i;
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

void bitonic_sort(int *array, int n) {
    int stage, bf_size, i;
    int indice_parceiro;

    for(bf_size = 2; bf_size <= n; bf_size *= 2) {
        for(stage = bf_size / 2; stage > 0; stage /= 2) {
            for(i = 0; i < n; i++) {
                    indice_parceiro = i ^ stage;
                    if(indice_parceiro > i) {
                        if(((i & bf_size) == 0 && array[i] > array[indice_parceiro]) ||
                           ((i & bf_size) != 0 && array[i] < array[indice_parceiro])) {
                            int temp = array[i];
                            array[i] = array[indice_parceiro];
                            array[indice_parceiro] = temp;                
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <array_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Array size must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    int size = pot_2(n);
    srand(time(NULL));
    int *array = generate_random_array(n, size);

    printf("Array before sorting: (first %d elements)\n", n);
    imprimi_array(array, n);

    bitonic_sort(array, size);

    printf("Array after sorting: (first %d elements)\n", n);
    imprimi_array(array, n);

    free(array);
    return EXIT_SUCCESS;
}