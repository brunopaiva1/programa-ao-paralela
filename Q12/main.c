/* 12. [Silva et al., 2022] Escreva um programa para calcular o produto escalar de dois vetores.
Utilize rotinas MPI_Send e MPI_Recv para comunicação entre os processos. Considere
cada vetor com N posições e divida a operação entre P processos distintos. Considere
que a divisão de N por P não tem resto.*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    int rank;
    int comm_sz;
    int n = 1000;
    int *A = NULL, *B = NULL;
    int *sub_A, *sub_B;
    int local_n;
    int local_dot = 0, global_dot;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if(n % comm_sz != 0) {
        if(rank == 0) {
            printf("Erro: (%d) não é divisível por (%d)\n", n, comm_sz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    local_n = n / comm_sz;
    sub_A = (int *)malloc(local_n * sizeof(int));
    sub_B = (int *)malloc(local_n * sizeof(int));

    if(rank==0){
        A = (int *)malloc(n * sizeof(int));
        B = (int *)malloc(n * sizeof(int));

        srand(time(NULL));
        for(int i = 0; i < n; i++){
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
        for (int i = 1; i < comm_sz; i++){
            MPI_Send(A + i * local_n, local_n, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(B + i * local_n, local_n, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
        
        for(int i = 0; i < local_n; i++){
            sub_A[i] = A[i];
            sub_B[i] = B[i];
        }
    } else{
        MPI_Recv(sub_A, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(sub_B, local_n, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for(int i = 0; i < local_n; i++){
        local_dot += sub_A[i] * sub_B[i];
    }

    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Produto escalar total: %d\n", global_dot);
        free(A);
        free(B);
    }

    free(sub_A);
    free(sub_B);

    MPI_Finalize();
    return 0;
}