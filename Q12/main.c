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
    int local_A, local_B;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    return 0;
}