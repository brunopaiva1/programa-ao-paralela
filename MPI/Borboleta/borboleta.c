/*9. [Pacheco and Malensek, 2022] Escreva um programa MPI que calcule uma soma global
usando uma borboleta. Primeiro, escreva seu programa para o caso especial em que
comm_sz é uma potência de dois. Modifique seu programa para que ele lide com qualquer
número de processos. */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void reducao_geral_borboleta(int *soma_local_processo, int rank, int comm_sz);

int main(int argc, char *argv[]) {
    int rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int soma_local_processo = rank + 1;
    reducao_geral_borboleta(&soma_local_processo, rank, comm_sz);

    printf("Processo %d -> Soma Global: %d\n", rank, soma_local_processo);

    MPI_Finalize();
    return 0;
}

void reducao_geral_borboleta(int *soma_local_processo, int rank, int comm_sz) {
    int passo_iteracao;
    int valor_recebido;
    int id_parceiro_comunicacao;

    int tam_pot2 = 1;
    while (tam_pot2 <= comm_sz) {
        tam_pot2 <<= 1;
    }
    tam_pot2 >>= 1;

    int contagem_processos_restantes = comm_sz - tam_pot2;

    if (rank >= tam_pot2) {
        MPI_Send(soma_local_processo, 1, MPI_INT, rank - tam_pot2, 0, MPI_COMM_WORLD);
    }
    
    if (rank < contagem_processos_restantes) {
        MPI_Recv(&valor_recebido, 1, MPI_INT, rank + tam_pot2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        (*soma_local_processo) += valor_recebido;
    }

    if (rank < tam_pot2) {
        for (passo_iteracao = 1; passo_iteracao < tam_pot2; passo_iteracao *= 2) {
            id_parceiro_comunicacao = rank ^ passo_iteracao;

            MPI_Sendrecv(soma_local_processo, 1, MPI_INT, id_parceiro_comunicacao, 0,
                         &valor_recebido, 1, MPI_INT, id_parceiro_comunicacao, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            (*soma_local_processo) += valor_recebido;
        }
    }

    if (rank < contagem_processos_restantes) {
        MPI_Send(soma_local_processo, 1, MPI_INT, rank + tam_pot2, 0, MPI_COMM_WORLD);
    }
    
    if (rank >= tam_pot2) {
        MPI_Recv(soma_local_processo, 1, MPI_INT, rank - tam_pot2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}