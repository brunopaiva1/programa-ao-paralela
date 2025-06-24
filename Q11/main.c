/*11. [Pacheco and Malensek, 2022] Escreva um programa que possa ser usado para determi-
nar o custo de alterar a distribuição de uma estrutura de dados distribuída. Quanto tempo
leva para mudar de uma distribuição em bloco de um vetor para uma distribuição cíclica?
Quanto tempo leva a redistribuição reversa?*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h> 

void gen_data(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }
}

void cyclic_distribution(int *local_data, int local_size, int my_rank, int comm_sz) {
    int local_flag = 1;
    for (int i = 0; i < local_size; i++) {
        if (local_data[i] != my_rank + i * comm_sz) {
            local_flag = 0;
            break;
        }
    }

    int global_flag_sum = 0;
    MPI_Reduce(&local_flag, &global_flag_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        if (global_flag_sum == comm_sz)
            printf("Verificação: Distribuição cíclica correta para todos os processos.\n");
        else
            printf("Verificação: Erro na distribuição cíclica em algum processo.\n");
    }
}

void block_distribuition(int *local_data, int local_size, int my_rank, int comm_sz, int size_data) {
    int chunk = size_data / comm_sz;
    int rest = size_data % comm_sz;
    int start_index = chunk * my_rank + (my_rank < rest ? my_rank : rest);
    
    int local_flag = 1; 
    for (int i = 0; i < local_size; i++) {
        if (local_data[i] != start_index + i) {
            local_flag = 0;
            break;
        }
    }

    int global_flag_sum = 0;
    MPI_Reduce(&local_flag, &global_flag_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        if (global_flag_sum == comm_sz)
            printf("Verificação: Distribuição em bloco correta para todos os processos.\n");
        else
            printf("Verificação: Erro na distribuição em bloco em algum processo.\n");
    }
}

int main(int argc, char *argv[]) {
    int my_rank, comm_sz, size_data;
    int *global_data = NULL, *local_data = NULL;
    int *send_counts = NULL, *displs = NULL;
    int chunk, rest, local_size_data;
    double start_time, end_time, elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Erro: Argumento ausente.\n");
            fprintf(stderr, "Uso: mpiexec -n <num_procs> %s <tamanho_do_array>\n", argv[0]);
            size_data = -1; 
        } else {
            size_data = atoi(argv[1]);
            if (size_data <= 0) {
                fprintf(stderr, "Erro: O tamanho do array deve ser um número positivo.\n");
                size_data = -1;
            }
        }
    }

    MPI_Bcast(&size_data, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    if (size_data <= 0) {
        MPI_Finalize();
        exit(1); 
    }
    chunk = size_data / comm_sz;
    rest = size_data % comm_sz;
    local_size_data = chunk + (my_rank < rest);

    local_data = (int *)malloc(local_size_data * sizeof(int));
    if (local_data == NULL) {
        fprintf(stderr, "Processo %d: Erro na alocação de local_data\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (my_rank == 0) {
        global_data = (int *)malloc(size_data * sizeof(int));
        send_counts = (int *)malloc(comm_sz * sizeof(int));
        displs = (int *)malloc(comm_sz * sizeof(int));

        if (global_data == NULL || send_counts == NULL || displs == NULL) {
            fprintf(stderr, "Processo %d: Erro na alocação de memória do processo raiz\n", my_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int offset = 0;
        for (int i = 0; i < comm_sz; i++) {
            send_counts[i] = chunk + (i < rest);
            displs[i] = offset;
            offset += send_counts[i];
        }
        gen_data(global_data, size_data); 
    }

    MPI_Scatterv(global_data, send_counts, displs, MPI_INT, 
                 local_data, local_size_data, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); 

    start_time = MPI_Wtime();

    MPI_Gatherv(local_data, local_size_data, MPI_INT, 
                global_data, send_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        int *temp_local_data_for_root = (int*)malloc(local_size_data * sizeof(int));
        if (temp_local_data_for_root == NULL) {
            fprintf(stderr, "Processo 0: Erro na alocação de temp_local_data_for_root\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < comm_sz; i++) {
            MPI_Datatype cyclic_type;
            int c = chunk + (i < rest ? 1 : 0); 
            MPI_Type_vector(c, 1, comm_sz, MPI_INT, &cyclic_type);
            MPI_Type_commit(&cyclic_type);

            if (i == 0) { 
                for (int j = 0; j < c; j++) {
                    temp_local_data_for_root[j] = global_data[j * comm_sz];
                }
            } else {
                MPI_Send(global_data + i, 1, cyclic_type, i, i, MPI_COMM_WORLD);
            }
            MPI_Type_free(&cyclic_type);
        }
        memcpy(local_data, temp_local_data_for_root, local_size_data * sizeof(int));
        free(temp_local_data_for_root);
    } else {
        MPI_Recv(local_data, local_size_data, MPI_INT, 0, my_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    end_time = MPI_Wtime();

    double global_start_time, global_end_time;
    MPI_Reduce(&start_time, &global_start_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end_time, &global_end_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        elapsed_time = global_end_time - global_start_time;
        printf("\nTempo para mudar de distribuicao em Bloco para Ciclica: %.5lfs\n", elapsed_time);
    }

    cyclic_distribution(local_data, local_size_data, my_rank, comm_sz);
    MPI_Barrier(MPI_COMM_WORLD);

    start_time = MPI_Wtime();

    if (my_rank == 0) {
        MPI_Datatype *recv_types = (MPI_Datatype *)malloc(comm_sz * sizeof(MPI_Datatype));
        if (recv_types == NULL) MPI_Abort(MPI_COMM_WORLD, 1);

        for (int i = 0; i < comm_sz; i++) {
            int current_rank_local_size = chunk + (i < rest);
            int *blocklengths = (int *)malloc(current_rank_local_size * sizeof(int));
            int *displacements = (int *)malloc(current_rank_local_size * sizeof(int));

            if (blocklengths == NULL || displacements == NULL) {
                fprintf(stderr, "Processo %d: Erro na alocação de blocklengths/displacements\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int j = 0; j < current_rank_local_size; j++) {
                blocklengths[j] = 1;
                displacements[j] = (i + j * comm_sz); 
            }
            MPI_Type_indexed(current_rank_local_size, blocklengths, displacements, MPI_INT, &recv_types[i]);
            MPI_Type_commit(&recv_types[i]);
            free(blocklengths);
            free(displacements);
        }

        for (int i = 0; i < local_size_data; i++) {
            global_data[i * comm_sz] = local_data[i];
        }

        for (int i = 1; i < comm_sz; i++) {
            MPI_Recv(global_data, 1, recv_types[i], i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Type_free(&recv_types[i]);
        }
        MPI_Type_free(&recv_types[0]);
        free(recv_types);

    } else {
        MPI_Send(local_data, local_size_data, MPI_INT, 0, my_rank, MPI_COMM_WORLD);
    }

    MPI_Scatterv(global_data, send_counts, displs, MPI_INT, 
                 local_data, local_size_data, MPI_INT, 0, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();

    MPI_Reduce(&start_time, &global_start_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end_time, &global_end_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        elapsed_time = global_end_time - global_start_time;
        printf("\nTempo para mudar de distribuicao Ciclica para Bloco: %.5lfs\n", elapsed_time);
    }

    block_distribuition(local_data, local_size_data, my_rank, comm_sz, size_data);

    if (my_rank == 0) {
        free(global_data);
        free(send_counts);
        free(displs);
    }
    free(local_data);

    MPI_Finalize();
    return 0;
}
