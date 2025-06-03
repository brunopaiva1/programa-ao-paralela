// 8. [Pacheco and Malensek, 2022] Escreva um programa MPI que calcule uma soma global
// estruturada em árvore. Primeiro, escreva seu programa para o caso especial em que
// comm_sz é uma potência de dois. Depois que esta versão estiver funcionando, modifique
// seu programa para que ele possa lidar com qualquer comm_sz.

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int comm_sz; 
    int rank; 
    int local_value = 0;
    int global_sum = 0; 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    local_value = rank;
    global_sum = local_value;

    int pd2 = 1;
    while (pd2 * 2 <= comm_sz) {
        pd2 *= 2;
    }

    if (rank >= pd2) {
        int partner = rank - pd2; 
        MPI_Send(&global_sum, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
    }
    else {
        if (rank + pd2 < comm_sz) {
            int partner = rank + pd2;
            int received_value;
            MPI_Recv(&received_value, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_sum += received_value; 
        }
    }

    if (rank < pd2) {
        for (int i = 0; i < (int)log2(pd2); i++) { 
            int partner;
            int mask = 1 << i;

            if ((rank & mask) == 0) { 
                partner = rank | mask;
                if (partner < pd2) { 
                    int received_value;
                    MPI_Recv(&received_value, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    global_sum += received_value;
                }
            } else { 
                partner = rank ^ mask;
                MPI_Send(&global_sum, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
                break;
            }
        }
    }

    if (rank == 0) {
        printf("Soma global final: %d\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}