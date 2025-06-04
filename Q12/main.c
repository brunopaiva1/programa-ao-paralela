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
    int n;
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
            printf("Erro: n (%d) não é divisível por comm_sz (%d)\n", n, comm_sz);
        }
        MPI_Finalize();
        return -1; 
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
    }

    return 0;
}