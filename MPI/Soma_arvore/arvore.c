/*8. [Pacheco and Malensek, 2022] Escreva um programa MPI que calcule uma soma global
estruturada em árvore. Primeiro, escreva seu programa para o caso especial em que
comm_sz é uma potência de dois. Depois que esta versão estiver funcionando, modifique
seu programa para que ele possa lidar com qualquer comm_sz.*/

#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, comm_sz, value, sum, partner;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

    value = rank + 1;

    int step = 1;
    while (step < comm_sz) {
        if ((rank & step) == 0) {
            partner = rank + step;
            if (partner < comm_sz) {
                MPI_Recv(&sum, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                value += sum;
            }
        } else {
            partner = rank - step;
            MPI_Send(&value, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            break;
        }
        step <<= 1;
    }

    if (rank == 0) {
        printf("Soma global final: %d\n", value);
    }

    MPI_Finalize();
    return 0;
}
