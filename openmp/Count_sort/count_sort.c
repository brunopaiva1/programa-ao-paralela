#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

void generate_random_array(int a[], int n, int num_threads) {
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        unsigned int seed = time(NULL) + tid;

        #pragma omp for
        for (int i = 0; i < n; i++) {
            a[i] = rand_r(&seed) % 100; 
        }
    }
}

void count_sort_parallel(int a[], int n, int num_threads) {
    int i, j, count;
    int* temp = malloc(n * sizeof(int));

    #pragma omp parallel for num_threads(num_threads) private(i, j, count) shared(a, temp, n) default(none)
    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++) {
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count++;
        }
        temp[count] = a[i];
    }

    memcpy(a, temp, n * sizeof(int));
    free(temp);
}

int main() {
    int n = 100000;
    int* a = malloc(n * sizeof(int));

    int num_threads;
    printf("Digite o número de threads: ");
    scanf("%d", &num_threads);

    printf("Gerando %d números aleatórios...\n", n);
    generate_random_array(a, n, num_threads);

    printf("Iniciando ordenação...\n");
    double start = omp_get_wtime();
    count_sort_parallel(a, n, num_threads);
    double end = omp_get_wtime();

    printf("Tempo de execução com %d threads: %f segundos\n", num_threads, end - start);
    printf("Ordenação finalizada.\n");

    printf("Primeiros 100000 elementos ordenados:\n");
    for (int i = 0; i < 100000; i++) {
        // printf("%d ", a[i]);
    }
    printf("\n");

    free(a);
    return 0;
}
