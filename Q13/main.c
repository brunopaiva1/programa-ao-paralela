/*13. [Silva et al., 2022] Considere um anel com num_procs processos. Escreva um programa
utilizando rotinas MPI_Send e MPI_Recv para comunicação entre os processos que faça
circular uma mensagem contendo um inteiro positivo ao longo desse canal. O processo
com ranque igual a 0 é quem inicia a transmissão e cada vez que a mensagem passa por
ele novamente o valor contido na mensagem deve ser decrementado de um até chegar
ao valor 0. Quando um processo receber a mensagem com valor 0 ele deverá passá-la
adiante e então terminar a sua execução. */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, num_procs;
    int message_value;
    int next_rank, prev_rank;
    int round_tag = 0; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Erro: Argumento ausente.\n");
            fprintf(stderr, "Uso: mpirun -np <num_procs> %s <valor_inicial>\n", argv[0]);
            message_value = -1;
        } else {
            message_value = atoi(argv[1]);
            if (message_value <= 0) {
                fprintf(stderr, "Erro: O valor inicial da mensagem deve ser um inteiro positivo.\n");
                message_value = -1; 
            }
        }
    }

    MPI_Bcast(&message_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (message_value < 0) {
        MPI_Finalize();
        return 1;
    }

    next_rank = (rank + 1) % num_procs;
    prev_rank = (rank - 1 + num_procs) % num_procs;

    while (1) {
        if (rank == 0) {
            if (round_tag == 0) { 
                printf("[Processo %d] INICIA com valor: %d (Tag: %d)\n", rank, message_value, round_tag);
            } else { 
                MPI_Recv(&message_value, 1, MPI_INT, prev_rank, round_tag - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("[Processo %d] RECEBEU valor %d (do %d, Tag: %d)\n", rank, message_value, prev_rank, round_tag - 1);
                message_value--; 
                printf("[Processo %d] DECREMENTOU para: %d\n", rank, message_value);
            }

            printf("[Processo %d] ENVIANDO valor %d para o processo %d (Tag: %d)\n", rank, message_value, next_rank, round_tag);
            MPI_Send(&message_value, 1, MPI_INT, next_rank, round_tag, MPI_COMM_WORLD);

            if (message_value == 0) {
                printf("[Processo %d] ULTIMO VALOR 0. Terminado.\n", rank);
                break; 
            }

        } else { 
            MPI_Recv(&message_value, 1, MPI_INT, prev_rank, round_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[Processo %d] RECEBEU valor %d do processo %d (Tag: %d)\n", rank, message_value, prev_rank, round_tag);

            printf("[Processo %d] ENVIANDO valor %d para o processo %d (Tag: %d)\n", rank, message_value, next_rank, round_tag);
            MPI_Send(&message_value, 1, MPI_INT, next_rank, round_tag, MPI_COMM_WORLD);

            if (message_value == 0) {
                printf("[Processo %d] ULTIMO VALOR 0. Terminado.\n", rank);
                break; 
            }
        }
        
        round_tag++;
    }

    MPI_Finalize();
    return 0;
}