/*10. [Pacheco and Malensek, 2022] Um merge sort paralelo começa com n/comm_sz chaves
atribuídas a cada processo. Ele termina com todas as chaves armazenadas no processo
0 em ordem de classificação. Para conseguir isso, ele usa a mesma comunicação estru-
turada em árvore que usamos para implementar uma soma global. No entanto, quando
um processo recebe as chaves de outro processo, ele mescla as novas chaves em sua
lista de chaves já ordenadas. Escreva um programa que implemente o merge sort para-
lelo. O processo 0 deve ler n e transmiti-lo para os outros processos. Cada processo deve
usar um gerador de números aleatórios para criar uma lista local de n/comm_sz inteiros.
Cada processo deve então classificar sua lista local e o processo 0 deve reunir e imprimir
as listas locais. Em seguida, os processos devem usar a comunicação estruturada em
árvore para mesclar a lista global no processo 0, que imprime o resultado.*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int compare_integers(const void *a, const void *b) {
    int int_a = *((int *)a);
    int int_b = *((int *)b);
    if (int_a == int_b) return 0;
    return (int_a < int_b) ? -1 : 1;
}

void generate_initial_data(int *array, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 10000;
    }
}

int main(int argc, char *argv[]) {
    int my_rank, total_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int n = 0;
    if (my_rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Uso: mpirun -np <procs> %s <num_elementos>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        if (my_rank == 0) fprintf(stderr, "Erro: O número de elementos deve ser positivo.\n");
        MPI_Finalize();
        return 1;
    }

    int local_n = n / total_processes;
    int resto = n % total_processes;
    if (my_rank < resto) {
        local_n++;
    }

    int *local_array = (int *)malloc(local_n * sizeof(int));

    if (my_rank == 0) {
        int *global_array = (int *)malloc(n * sizeof(int));
        generate_initial_data(global_array, n);

        int *send_counts = (int *)malloc(total_processes * sizeof(int));
        int *deslocamento = (int *)malloc(total_processes * sizeof(int));
        int deslocamento_atual = 0;

        for (int i = 0; i < total_processes; i++) {
            send_counts[i] = (n / total_processes) + (i < resto);
            deslocamento[i] = deslocamento_atual;
            deslocamento_atual += send_counts[i];
        }

        MPI_Scatterv(global_array, send_counts, deslocamento, MPI_INT,
                     local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        free(global_array);
        free(send_counts);
        free(deslocamento);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    }


    double start_time = MPI_Wtime();
    qsort(local_array, local_n, sizeof(int), compare_integers);

    for (int step = 1; step < total_processes; step *= 2) {
        if ((my_rank % (2 * step)) == 0) { 
            int partner = my_rank + step;
            if (partner < total_processes) {
                int received_n;
                MPI_Recv(&received_n, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int *received_array = (int *)malloc(received_n * sizeof(int));
                MPI_Recv(received_array, received_n, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int new_size = local_n + received_n;
                int *merged_array = (int *)malloc(new_size * sizeof(int));

                int i = 0, j = 0, k = 0;
                while (i < local_n && j < received_n) {
                    if (local_array[i] <= received_array[j]) {
                        merged_array[k++] = local_array[i++];
                    } else {
                        merged_array[k++] = received_array[j++];
                    }
                }
                while (i < local_n) merged_array[k++] = local_array[i++];
                while (j < received_n) merged_array[k++] = received_array[j++];

                free(local_array);
                free(received_array);
                local_array = merged_array;
                local_n = new_size;
            }
        } else { 
            int partner = my_rank - step;
            MPI_Send(&local_n, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(local_array, local_n, MPI_INT, partner, 0, MPI_COMM_WORLD);
            break;
        }
    }
    double end_time = MPI_Wtime();

    if (my_rank == 0) {
        printf("Tempo total de ordenação e fusão: %f segundos\n", end_time - start_time);
        printf("Processos: %d, Elementos: %d\n", total_processes, n);

        int sorted = 1;
        for (int i = 0; i < n - 1; i++) {
            if (local_array[i] > local_array[i + 1]) {
                sorted = 0;
                break;
            }
        }
        if (sorted) {
            printf("Sucesso: O array foi ordenado corretamente.\n");
        } else {
            printf("Erro: O array não foi ordenado corretamente.\n");
        }
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}