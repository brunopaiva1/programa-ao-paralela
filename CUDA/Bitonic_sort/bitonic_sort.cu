#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <cuda_runtime.h> // Cabeçalho principal da CUDA

// --- Funções Auxiliares (sem alteração) ---

int pot_2(int n) {
    int i = 1;
    while (i < n) {
        i *= 2;
    }
    return i;
}

int *generate_random_array(int n, int size) {
    int *array = (int *)malloc(size * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        array[i] = rand() % (INT_MAX/100000);
    }
    for(int i = n; i < size; i++) {
        array[i] = INT_MAX;
    }
    return array;
}

void imprimi_array(int *array, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

// --- Kernel CUDA ---

/**
 * @brief Executa uma única etapa de comparação e troca da ordenação bitônica.
 * @param array Ponteiro para o array na memória da GPU.
 * @param stage A distância de comparação para a etapa atual.
 * @param bf_size O tamanho da subsequência bitônica sendo mesclada.
 * @param size O tamanho total do array (para verificação de limites).
 */
__global__ void bitonic_sort_kernel(int *array, int stage, int bf_size, int size) {
    // Calcula o índice global do elemento que esta thread irá processar
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Garante que a thread não acesse memória fora do array
    if (i >= size) {
        return;
    }

    // Lógica de comparação e troca (idêntica à sua versão sequencial)
    int indice_parceiro = i ^ stage;
    if (indice_parceiro > i) {
        // Determina a direção da ordenação (crescente ou decrescente)
        bool is_ascending = ((i & bf_size) == 0);
        
        if ((is_ascending && array[i] > array[indice_parceiro]) ||
            (!is_ascending && array[i] < array[indice_parceiro])) {
            
            int temp = array[i];
            array[i] = array[indice_parceiro];
            array[indice_parceiro] = temp;
        }
    }
}


// --- Função Principal ---

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s <tamanho_array>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "O tamanho do array deve ser um inteiro positivo.\n");
        return EXIT_FAILURE;
    }

    // --- Preparação dos Dados (Host) ---
    int size = pot_2(n);
    size_t size_bytes = size * sizeof(int);
    int *h_array = generate_random_array(n, size);

    // --- Alocação e Cópia para a GPU ---
    int *d_array;
    cudaMalloc(&d_array, size_bytes);
    cudaMemcpy(d_array, h_array, size_bytes, cudaMemcpyHostToDevice);

    printf("Array antes da ordenação (primeiros %d elementos de %d):\n", n, size);
    imprimi_array(h_array, n);

    // --- Execução da Ordenação na GPU ---
    const int THREADS_PER_BLOCK = 256;
    const int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Laços de controle que permanecem no host
    for (int bf_size = 2; bf_size <= size; bf_size *= 2) {
        for (int stage = bf_size / 2; stage > 0; stage /= 2) {
            // Lança o kernel para executar a etapa em paralelo
            bitonic_sort_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_array, stage, bf_size, size);
            
            // Sincroniza para garantir que a etapa termine antes de iniciar a próxima
            cudaDeviceSynchronize();
        }
    }

    // --- Cópia do Resultado de Volta para o Host ---
    cudaMemcpy(h_array, d_array, size_bytes, cudaMemcpyDeviceToHost);

    printf("\nArray depois da ordenação (primeiros %d elementos):\n", n);
    imprimi_array(h_array, n);

    // --- Limpeza ---
    free(h_array);
    cudaFree(d_array);
    
    return EXIT_SUCCESS;
}