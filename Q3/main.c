#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>   

#define N 10000

#define EPSILON 1e-9

double A_mat[N][N];
double b_vec[N];
double x_parallel[N];
double x_serial[N];  

void initialize_system() {
    int i, j;
    for (i = 0; i < N; i++) {
        b_vec[i] = (double)(i + 1.0); 
        x_parallel[i] = 0.0;          
        x_serial[i] = 0.0;           
        for (j = 0; j < N; j++) {
            if (j >= i) { 
                A_mat[i][j] = (double)(i + j + 1.0);
                if (i == j) {
                    A_mat[i][j] += 10.0;
                }
            } else {
                A_mat[i][j] = 0.0;
            }
        }
    }
}

void back_substitution_row_oriented_serial(double A[N][N], double b[N], double x[N]) {
    int lin, col;
    for (lin = N - 1; lin >= 0; lin--) {
        x[lin] = b[lin];
        for (col = lin + 1; col < N; col++)
            x[lin] -= A[lin][col] * x[col];
        x[lin] /= A[lin][lin];
    }
}

void back_substitution_column_oriented_serial(double A[N][N], double b[N], double x[N]) {
    int i, col, lin;
    for (i = 0; i < N; i++)
        x[i] = b[i];
    for (col = N - 1; col >= 0; col--) {
        x[col] /= A[col][col];
        for (lin = 0; lin < col; lin++)
            x[lin] -= A[lin][col] * x[col];
    }
}

void back_substitution_row_oriented_parallel_inner(double A[N][N], double b[N], double x[N]) {
    int lin, col;
    for (lin = N - 1; lin >= 0; lin--) {
        double current_b = b[lin];
        double terms_to_subtract = 0.0;
        #pragma omp parallel for reduction(+:terms_to_subtract)
        for (col = lin + 1; col < N; col++) {
            terms_to_subtract += A[lin][col] * x[col];
        }
        x[lin] = (current_b - terms_to_subtract) / A[lin][lin];
    }
}

void back_substitution_column_oriented_parallel_initial_loop(double A[N][N], double b[N], double x[N]) {
    int i, col, lin;
    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        x[i] = b[i];
    }
    for (col = N - 1; col >= 0; col--) {
        x[col] /= A[col][col];
        for (lin = 0; lin < col; lin++)
            x[lin] -= A[lin][col] * x[col];
    }
}

void back_substitution_column_oriented_parallel_inner(double A[N][N], double b[N], double x[N]) {
    int i, col, lin;
    for (i = 0; i < N; i++) {
        x[i] = b[i];
    }
    for (col = N - 1; col >= 0; col--) {
        #pragma omp single
        x[col] /= A[col][col];
        #pragma omp parallel for
        for (lin = 0; lin < col; lin++) {
            x[lin] -= A[lin][col] * x[col];
        }
    }
}

int compare_results(double* result_parallel, double* result_serial, int size, const char* test_name) {
    int errors = 0;
    for (int i = 0; i < size; i++) {
        if (fabs(result_parallel[i] - result_serial[i]) > EPSILON) {
            if (errors < 5) { 
                printf("ERRO no %s: x_parallel[%d] = %f, x_serial[%d] = %f\n",
                       test_name, i, result_parallel[i], i, result_serial[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("RESULTADO do %s: CORRETO (paralelo e serial são iguais dentro da tolerância).\n", test_name);
        return 1;
    } else {
        printf("RESULTADO do %s: INCORRETO (%d erros encontrados).\n", test_name, errors);
        return 0;
    }
}

int main() {
    initialize_system();

    double start_time, end_time;

    printf("\n--- Teste de Corretude: Algoritmo Orientado a Linhas (Laco Interno Paralelo) ---\n");

    memcpy(x_serial, b_vec, N * sizeof(double));
    start_time = omp_get_wtime();
    back_substitution_row_oriented_serial(A_mat, b_vec, x_serial);
    end_time = omp_get_wtime();
    printf("Tempo Serial: %f segundos\n", end_time - start_time);

    memcpy(x_parallel, b_vec, N * sizeof(double)); 
    start_time = omp_get_wtime();
    back_substitution_row_oriented_parallel_inner(A_mat, b_vec, x_parallel);
    end_time = omp_get_wtime();
    printf("Tempo Paralelo: %f segundos\n", end_time - start_time);

    compare_results(x_parallel, x_serial, N, "Orientado a Linhas (Laco Interno Paralelo)");
    printf("\n--- Teste de Corretude: Algoritmo Orientado a Colunas (Primeiro Laco Paralelo) ---\n");

    memcpy(x_serial, b_vec, N * sizeof(double));
    start_time = omp_get_wtime();
    back_substitution_column_oriented_serial(A_mat, b_vec, x_serial);
    end_time = omp_get_wtime();
    printf("Tempo Serial: %f segundos\n", end_time - start_time);

    memcpy(x_parallel, b_vec, N * sizeof(double));
    start_time = omp_get_wtime();
    back_substitution_column_oriented_parallel_initial_loop(A_mat, b_vec, x_parallel);
    end_time = omp_get_wtime();
    printf("Tempo Paralelo: %f segundos\n", end_time - start_time);

    compare_results(x_parallel, x_serial, N, "Orientado a Colunas (Primeiro Laco Paralelo)");

    printf("\n--- Teste de Corretude: Algoritmo Orientado a Colunas (Laco Interno Paralelo) ---\n");

    memcpy(x_serial, b_vec, N * sizeof(double));
    start_time = omp_get_wtime();
    back_substitution_column_oriented_serial(A_mat, b_vec, x_serial);
    end_time = omp_get_wtime();
    printf("Tempo Serial: %f segundos\n", end_time - start_time);

    memcpy(x_parallel, b_vec, N * sizeof(double));
    start_time = omp_get_wtime();
    back_substitution_column_oriented_parallel_inner(A_mat, b_vec, x_parallel);
    end_time = omp_get_wtime();
    printf("Tempo Paralelo: %f segundos\n", end_time - start_time);

    compare_results(x_parallel, x_serial, N, "Orientado a Colunas (Laco Interno Paralelo)");

    return 0;
}