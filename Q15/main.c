/*15. [Silva et al., 2022] A Conjectura de Collatz foi desenvolvida pelo matemático alemão
Lothar Collatz. Nela, escolhendo-se um número natural inicial N , onde N > 0, os se-
guintes critérios serão obedecidos: Se N for par o seu sucessor será a metade e se N
for ímpar o seu sucessor será o triplo mais um, gerando então um novo número. Esse
processo repete-se até que eventualmente se atinja o número 1.
Use o modelo “saco de tarefa” para resolver esse problema em paralelo e verificar se
os números inteiros no intervalo de 1 a 100.000.000 atendem a esses critérios e o maior
número de passos necessários para chegar até o número 1. Atribua a um dos processos
o papel de “mestre” (gerencia a distribuição de tarefas) e aos demais processos o papel
de “trabalhadores” (executam a tarefa de avaliar se um número está dentro da conjec-
tura). Utilize mensagens de envio e recepção não bloqueantes de forma a paralelizar a
computação com o envio de mensagens.*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define RANGE 10
#define TASK_SIZE 2

int collatz_steps(unsigned int n){
    int steps = 0; 
    while(n != 1){
        if(n % 2 == 0) {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        steps++;
    }
    return steps;
}

int main(int argc, char *argv[]){
    int rank, comm_size;
    int local_max_steps;
    MPI_Request request;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if(rank == 0){
        int next_task = 1;
        int works_done = 0;
        int task_data[2];
        int max_steps_recv;

        for (int i = 1; i < comm_size; i++) {
            if (next_task <= RANGE) {
            task_data[0] = next_task;
            task_data[1] = next_task + TASK_SIZE - 1;
            next_task += TASK_SIZE;
            MPI_Send(task_data, 2, MPI_INT, i, 0, MPI_COMM_WORLD);            }
        }
        while (works_done < comm_size - 1) {
            MPI_Recv(&max_steps_recv, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        }
        if (next_task <= RANGE) {
                task_data[0] = next_task;
                task_data[1] = next_task + TASK_SIZE - 1;
                next_task += TASK_SIZE;
                MPI_Send(task_data, 2, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            } else {
                task_data[0] = 0;
                task_data[1] = 0;
                MPI_Send(task_data, 2, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                works_done++;
            }

        printf("Maior número de passos: %d\n", local_max_steps);
    } else {
        int task_data[2];
        int max_steps = 0;
        while (1) {
            MPI_Recv(task_data, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            if (task_data[0] == 0 && task_data[1] == 0)
                break;

            for (int i = task_data[0]; i <= task_data[1] && i <= RANGE; i++) {
                int steps = collatz_steps(i);
                if (steps > max_steps)
                    max_steps = steps;
            }

            MPI_Send(&max_steps, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
