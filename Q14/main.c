/* 14. [Silva et al., 2022] Escreva uma rotina para calcular o produto escalar de dois vetores,
utilizando rotinas de comunicação coletiva do MPI para envio do vetor inicial e recepção
dos resultados parciais pelo processo raiz. Considere cada vetor com N posições e divida
a operação entre P processos distintos. Considere que a divisão de N por P não tem
resto. Assuma que o valor de N não é conhecido e deve ser enviado pelo processo raiz
para os demais processos com uso de uma rotina comunicação coletiva. */

#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
    int rank, comm_sz;
    int n;
    int *A = NULL, *B = NULL;
    int *sub_A, *sub_B;
    int local_n;
    int local_dot, global_dot;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if(rank == 0){
        n = 1000;
        if(n % comm_sz != 0){
            printf("ERRO %d não é divisivel por %d\n", n, comm_sz);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_n = n / comm_sz;
    sub_A = (int *)malloc(local_n * sizeof(int));
    sub_B = (int *)malloc(local_n * sizeof(int));

    if(rank == 0){
        A = (int *)malloc(n * sizeof(int));
        B = (int *)malloc(n * sizeof(int));

        srand(time(NULL));
        for(int i = 0; i < n; i++){
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
        start_time = MPI_Wtime();
    }

    MPI_Scatter(A, local_n, MPI_INT, sub_A, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, local_n, MPI_INT, sub_B, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i = 0; i < local_n; i++){
        local_dot += sub_A[i] * sub_B[i];
    }

    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        end_time = MPI_Wtime();
        printf("Produto escalar total: %d\n", global_dot);
        printf("Tempo de execução: %f segundos\n", end_time - start_time);
    }

    free(sub_A);
    free(sub_B);
    if(rank == 0){
        free(A);
        free(B);
    }

    MPI_Finalize();
    return 0;
}