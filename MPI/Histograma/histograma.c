/* Use o MPI para implementar o programa de histograma discutido na Seção 2.7.1. Faça
 com que o processo 0 leia os dados de entrada e os distribua entre os processos. Faça
 também que o processo 0 imprima o histograma. */

// mpicc -o saida1 parallel.c 
// mpiexec -np 4 ./saida1

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define BIN_COUNT 6
#define DATA_COUNT 5000  

void find_bin(float value, float min_val, float max_val, int bin_count, int* bin_index) {
    float bin_width = (max_val - min_val) / bin_count;
    *bin_index = (int)((value - min_val) / bin_width);
    if (*bin_index == bin_count){
        *bin_index = bin_count - 1;
    } 
}

int main(int argc, char* argv[]) {
    int rank, size;
    float min_val = 0.0, max_val = 6.0;
    int bin_count = BIN_COUNT;
    int* global_bin_counts = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base = DATA_COUNT / size;
    int resto = DATA_COUNT % size;

    int* data_chunks = malloc(size * sizeof(int));
    int* start_indices = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        data_chunks[i] = base + (i < resto ? 1 : 0);
    }

    start_indices[0] = 0;
    for (int i = 1; i < size; i++) {
        start_indices[i] = start_indices[i - 1] + data_chunks[i - 1];
    }

    float* local_data = malloc(data_chunks[rank] * sizeof(float));

    if (rank == 0) {
        float* data = malloc(DATA_COUNT * sizeof(float));

        srand(time(NULL));
        for (int i = 0; i < DATA_COUNT; i++) {
            float r = ((float) rand() / RAND_MAX);
            data[i] = min_val + r * (max_val - min_val); 
        }

        printf("Gerados %d valores float aleatórios no intervalo [%.1f, %.1f].\n", DATA_COUNT, min_val, max_val);

        MPI_Scatterv(data, data_chunks, start_indices, MPI_FLOAT,
                     local_data, data_chunks[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

        free(data);
    } else {
        MPI_Scatterv(NULL, data_chunks, start_indices, MPI_FLOAT,
                     local_data, data_chunks[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    int* local_bin_counts = calloc(bin_count, sizeof(int));
    for (int i = 0; i < data_chunks[rank]; i++) {
        int bin_index;
        find_bin(local_data[i], min_val, max_val, bin_count, &bin_index);
        local_bin_counts[bin_index]++;
    }

    if (rank == 0) {
        global_bin_counts = calloc(bin_count, sizeof(int));
    }

    MPI_Reduce(local_bin_counts, global_bin_counts, bin_count, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nHistograma final:\n");
        for (int i = 0; i < bin_count; i++) {
            printf("Bin %d (%.1f - %.1f): %d\n", i,
                   min_val + i * (max_val - min_val) / bin_count,
                   min_val + (i + 1) * (max_val - min_val) / bin_count,
                   global_bin_counts[i]);
        }
        free(global_bin_counts);
    }

    free(local_data);
    free(local_bin_counts);
    free(data_chunks);
    free(start_indices);

    MPI_Finalize();
    return 0;
}
