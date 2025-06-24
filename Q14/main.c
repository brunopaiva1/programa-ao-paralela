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

    if(n % comm_sz != 0){
        if(rank == 0){
            printf("ERRO %d não é divisivel por %d", &n, &comm_sz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    local_n = n / comm_sz;
    sub_A = (int *)malloc(local_n * sizeof(int));
    sub_B = (int *)malloc(local_n * sizeof(int));

    return 0;
}