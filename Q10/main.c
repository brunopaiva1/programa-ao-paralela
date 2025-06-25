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

void merge_local(int arr[], int left, int middle, int right) {
    int i, j, k;
    int n1 = middle - left + 1;
    int n2 = right - middle;

    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    if (L == NULL || R == NULL) {
        fprintf(stderr, "Erro: Falha ao alocar memória temporária para merge local.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (i = 0; i < n1; i++) L[i] = arr[left + i];
    for (j = 0; j < n2; j++) R[j] = arr[middle + 1 + j];

    i = 0; j = 0; k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void perform_local_merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int middle = left + (right - left) / 2;
        perform_local_merge_sort(arr, left, middle);
        perform_local_merge_sort(arr, middle + 1, right);
        merge_local(arr, left, middle, right);
    }
}

void generate_global_data(int *array, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 10000; 
    }
}

int main(int argc, char *argv[]) {
    int my_rank, total_processes;
    int total_elements_N = 0;
    int local_elements_count;
    int *local_data_array = NULL;
    int *global_data_source = NULL; 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Uso: mpirun -np <num_procs> %s <numero_total_de_elementos>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        total_elements_N = atoi(argv[1]);
        if (total_elements_N <= 0) {
            fprintf(stderr, "Erro: O número total de elementos deve ser um inteiro positivo.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&total_elements_N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_base = total_elements_N / total_processes;
    int remainder = total_elements_N % total_processes;
    local_elements_count = chunk_base + (my_rank < remainder);
    
    local_data_array = (int *)malloc(local_elements_count * sizeof(int));
    if (local_data_array == NULL) {
        fprintf(stderr, "Processo %d: Erro de alocação para dados locais.\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (my_rank == 0) {
        global_data_source = (int *)malloc(total_elements_N * sizeof(int));
        if (global_data_source == NULL) {
            fprintf(stderr, "P0: Erro de alocação para dados globais de origem.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        generate_global_data(global_data_source, total_elements_N);

        int *send_counts = (int *)malloc(total_processes * sizeof(int));
        int *displacements = (int *)malloc(total_processes * sizeof(int));
        int current_offset = 0;
        for (int i = 0; i < total_processes; i++) {
            send_counts[i] = chunk_base + (i < remainder);
            displacements[i] = current_offset;
            current_offset += send_counts[i];
        }

        MPI_Scatterv(global_data_source, send_counts, displacements, MPI_INT, 
                     local_data_array, local_elements_count, MPI_INT, 0, MPI_COMM_WORLD);
        
        free(global_data_source);
        free(send_counts);
        free(displacements);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     local_data_array, local_elements_count, MPI_INT, 0, MPI_COMM_WORLD);
    }

    perform_local_merge_sort(local_data_array, 0, local_elements_count - 1);

    double start_time = MPI_Wtime();

    int current_jump_size;
    for (current_jump_size = 1; current_jump_size < total_processes; current_jump_size <<= 1) {
        int partner_rank;

        if ((my_rank & current_jump_size) == 0) { 
            partner_rank = my_rank + current_jump_size;
            if (partner_rank < total_processes) { 
                int received_elements_count;
                MPI_Recv(&received_elements_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                int *received_data = (int *)malloc(received_elements_count * sizeof(int));
                if (received_data == NULL) {
                    fprintf(stderr, "P%d: Erro de alocação para dados recebidos.\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                MPI_Recv(received_data, received_elements_count, MPI_INT, partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int *merged_list = (int *)malloc((local_elements_count + received_elements_count) * sizeof(int));
                if (merged_list == NULL) {
                    fprintf(stderr, "P%d: Erro de alocação para lista mesclada.\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                int i = 0, j = 0, k = 0;
                while (i < local_elements_count && j < received_elements_count) {
                    if (local_data_array[i] <= received_data[j]) merged_list[k++] = local_data_array[i++];
                    else merged_list[k++] = received_data[j++];
                }
                while (i < local_elements_count) merged_list[k++] = local_data_array[i++];
                while (j < received_elements_count) merged_list[k++] = received_data[j++];

                free(local_data_array); 
                free(received_data);
                local_data_array = merged_list; 
                local_elements_count += received_elements_count;
            }
        } else { 
            partner_rank = my_rank - current_jump_size;
            MPI_Send(&local_elements_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
            MPI_Send(local_data_array, local_elements_count, MPI_INT, partner_rank, 1, MPI_COMM_WORLD);
            
            free(local_data_array); 
            local_data_array = NULL; 
            local_elements_count = 0; 
            break; 
        }
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if (my_rank == 0) {
        printf("----------------------------------------\n");
        printf("Tempo total de ordenação e fusão: %f segundos com %d processos e %d elementos.\n", 
               elapsed_time, total_processes, total_elements_N);
        
        int sorted_check = 1;
        for (int i = 0; i < local_elements_count - 1; i++) {
            if (local_data_array[i] > local_data_array[i + 1]) {
                sorted_check = 0;
                break;
            }
        }

        if (total_elements_N < 200) { 
            printf("Array Final Ordenado (no processo 0):\n[");
            for (int i = 0; i < local_elements_count; i++) {
                printf("%d%s", local_data_array[i], (i == local_elements_count - 1) ? "" : ", ");
            }
            printf("]\n");
        }

        if (sorted_check) {
            printf("Sucesso: O array foi ordenado corretamente.\n");
        } else {
            printf("Erro: O array não foi ordenado corretamente.\n");
        }
        printf("----------------------------------------\n");
    }

    if (local_data_array != NULL) {
        free(local_data_array);
    }
    
    MPI_Finalize();
    return 0;
}