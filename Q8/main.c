// 8. [Pacheco and Malensek, 2022] Escreva um programa MPI que calcule uma soma global
// estruturada em árvore. Primeiro, escreva seu programa para o caso especial em que
// comm_sz é uma potência de dois. Depois que esta versão estiver funcionando, modifique
// seu programa para que ele possa lidar com qualquer comm_sz.

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char argv){
    int rank;
    int comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    return 0;
}