#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_QUEUE 100
#define MAX_LINHA 512
#define NUM_CONSUMIDORES 2

char* fila[MAX_QUEUE];
int inicio = 0, fim = 0, contador = 0;

void enfileirar(char* linha) {
    int inserido = 0;
    while (!inserido) {
        #pragma omp critical
        {
            if (contador < MAX_QUEUE) {
                fila[fim] = strdup(linha);
                fim = (fim + 1) % MAX_QUEUE;
                contador++;
                inserido = 1;
            }
        }
    }
}

char* desenfileirar() {
    char* linha = NULL;
    int retirado = 0;
    while (!retirado) {
        #pragma omp critical
        {
            if (contador > 0) {
                linha = fila[inicio];
                inicio = (inicio + 1) % MAX_QUEUE;
                contador--;
                retirado = 1;
            }
        }
    }
    return linha;
}

void consumir() {
    while (1) {
        char* linha = desenfileirar();
        if (strcmp(linha, "__EOF__") == 0) {
            free(linha);
            break;
        }
        char* token = strtok(linha, " \t\n");
        while (token != NULL) {
            printf("Token: %s\n", token);
            token = strtok(NULL, " \t\n");
        }
        free(linha);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s arquivo1.txt [arquivo2.txt ...]\n", argv[0]);
        return 1;
    }

    int num_produtores = argc - 1;

    #pragma omp parallel num_threads(num_produtores + NUM_CONSUMIDORES)
    {
        int id = omp_get_thread_num();

        if (id < num_produtores) {
            FILE* f = fopen(argv[id + 1], "r");
            if (!f) {
                fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[id + 1]);
                exit(1);
            }

            char linha[MAX_LINHA];
            while (fgets(linha, sizeof(linha), f)) {
                enfileirar(linha);
            }
            fclose(f);
            #pragma omp barrier
            #pragma omp single
            {
                for (int i = 0; i < NUM_CONSUMIDORES; i++) {
                    enfileirar(strdup("__EOF__"));
                }
            }

        } else if (id < num_produtores + NUM_CONSUMIDORES) {
            consumir();
        }
    }

    return 0;
}
