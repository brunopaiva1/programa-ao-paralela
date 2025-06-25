#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void butterfly_all_reduce(int *process_local_sum, int current_process_rank, int total_processes);

int main(int argc, char *argv[]) {
    int current_process_rank, total_processes;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process_rank);

    int process_local_sum = current_process_rank + 1;
    butterfly_all_reduce(&process_local_sum, current_process_rank, total_processes);

    printf("Processo %d -> Soma Global: %d\n", current_process_rank, process_local_sum);

    MPI_Finalize();
    return 0;
}

void butterfly_all_reduce(int *process_local_sum, int current_process_rank, int total_processes) {
    int iteration_step;
    int received_value;
    int communication_partner_rank;

    int largest_power_of_two_group_size = 1;
    while (largest_power_of_two_group_size < total_processes) {
        largest_power_of_two_group_size <<= 1;
    }
    largest_power_of_two_group_size >>= 1;

    int remaining_processes_count = total_processes - largest_power_of_two_group_size;

    if (current_process_rank >= largest_power_of_two_group_size) {
        MPI_Send(process_local_sum, 1, MPI_INT, current_process_rank - largest_power_of_two_group_size, 0, MPI_COMM_WORLD);
    }
    
    if (current_process_rank < remaining_processes_count) {
        MPI_Recv(&received_value, 1, MPI_INT, current_process_rank + largest_power_of_two_group_size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        (*process_local_sum) += received_value;
    }

    if (current_process_rank < largest_power_of_two_group_size) {
        for (iteration_step = 1; iteration_step < largest_power_of_two_group_size; iteration_step *= 2) {
            communication_partner_rank = current_process_rank ^ iteration_step;

            MPI_Sendrecv(process_local_sum, 1, MPI_INT, communication_partner_rank, 0,
                         &received_value, 1, MPI_INT, communication_partner_rank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            (*process_local_sum) += received_value;
        }
    }

    if (current_process_rank < remaining_processes_count) {
        MPI_Send(process_local_sum, 1, MPI_INT, current_process_rank + largest_power_of_two_group_size, 0, MPI_COMM_WORLD);
    }
    
    if (current_process_rank >= largest_power_of_two_group_size) {
        MPI_Recv(process_local_sum, 1, MPI_INT, current_process_rank - largest_power_of_two_group_size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}