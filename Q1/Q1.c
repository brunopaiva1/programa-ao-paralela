#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double EstPi(int thread_count, long long int num_lancamentos){
    int lancamentos;
    double x, y, dist_2;
    long long int qtd_no_circulo = 0;
    srand(time(NULL));

    #pragma omp parallel for num_threads(thread_count) reduction(+:qtd_no_circulo) \
    default(none) private(x, y, dist_2, lancamentos) shared(num_lancamentos)
    for(lancamentos = 0; lancamentos < num_lancamentos; lancamentos++){
        unsigned int seed = (unsigned int) time(NULL) ^ (omp_get_thread_num() + lancamentos);
        x = ((double)rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        y = ((double)rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        dist_2 = (x*x)+(y*y);
        if (dist_2 <= 1) qtd_no_circulo++;
    }
    return (4 * qtd_no_circulo)/((double) num_lancamentos);
}

void Usage(const char* prog_nome) {
    fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_nome);
    fprintf(stderr, " thread_count é o número de thread >= 1\n");
    fprintf(stderr, " n é o número de jogadas a serem realizadas >= 1\n");
    exit(0);
}

int main(int argc, char const *argv[])
{
    double inicio, fim;

    inicio = omp_get_wtime();

    int thread_count;
    long long int num_lancamentos;
    if (argc != 3) Usage(argv[0]);
    thread_count = strtol(argv[1], NULL, 10);
    num_lancamentos = strtol(argv[2], NULL, 10);
    printf("Valor da estimativa: %lf\n", EstPi(thread_count, num_lancamentos));

    fim = omp_get_wtime();
    printf("Tempo de execução: %f segundos \n", fim - inicio);
    
    return 0;
}
