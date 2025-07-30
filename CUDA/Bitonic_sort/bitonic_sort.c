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

int *generate_random_array(int n) {
    int *array = (int *)malloc(n * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        array[i] = rand() % (INT_MAX/10000000);
    }

    for(int i = n; i < pot_2(n); i++) {
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

}