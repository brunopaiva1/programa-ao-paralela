/*7. [Pacheco and Malensek, 2022] Escreva um programa MPI que use um método de Monte
Carlo para estimar π (Seção 1, questão 1). O processo 0 deve ler o número total de lan-
çamentos e transmití-lo aos outros processos. Use o MPI_Reduce para encontrar a soma
global da variável local qtd_no_circulo e peça ao processo 0 para imprimir o resultado.*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    int comm_sz;
    int rank;
    long long int total_num_lancamentos;
    long long int local_num_lancamentos;
    long long int local_qtd_no_circulo = 0;
    long long int global_qtd_no_circulo;
    double pi_estimado;
    double start_time, end_time;

    MPI_Init(&argc,  &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if(rank == 0){
        if (argc != 2) {
            fprintf(stderr, "Uso: %s <num_lancamentos>\n", argv[0]);
            fprintf(stderr, "num_lancamentos é o número total de jogadas a serem realizadas >= 1\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        total_num_lancamentos = strtoll(argv[1], NULL, 10);
        if (total_num_lancamentos < 1) {
            fprintf(stderr, "O número de lançamentos deve ser maior ou igual a 1.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Número total de lançamentos: %lld\n", total_num_lancamentos);
        start_time = MPI_Wtime();
    }

    MPI_Bcast(&total_num_lancamentos, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    
    local_num_lancamentos = total_num_lancamentos / comm_sz;

    if (rank == 0) {
        local_num_lancamentos += (total_num_lancamentos % comm_sz);
    }

    srand((unsigned int)time(NULL) + rank);
    
    for (long long int i = 0; i < local_num_lancamentos; i++) {
        double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double dist_2 = (x * x) + (y * y);
        if (dist_2 <= 1.0) {
            local_qtd_no_circulo++;
        }
    }
    
    MPI_Reduce(&local_qtd_no_circulo, &global_qtd_no_circulo, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        pi_estimado = (4.0 * global_qtd_no_circulo) / ((double)total_num_lancamentos);
        end_time = MPI_Wtime();

        printf("Estimativa de Pi: %lf\n", pi_estimado);
        printf("Tempo de execução: %lf segundos\n", end_time - start_time);
    }

    MPI_Finalize();

    return 0;
}