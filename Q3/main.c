#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 1000

double A[N][N];
double b[N];
double x_lin[N];
double x_col[N];

void inicialize_system(double A_mat[N][N], double b_vec[N], double x_vec_lin[N], double x_vec_col[N]){
    int i, j;
    for(i = 0; i < N; i++){
        b_vec[i] = (double)(i + 1.0);
        x_vec_lin[i] = 0.0;
        x_vec_col[i] = 0.0;
        for(j = 0; j < N; j++){
            if(j >= i){
                A_mat[i][j] = (double)(i + j + 1.0);
                if(i == j){
                    A[i][j] += 10.0;
                }
            } else {
                A_mat[i][j] = 0.0;
            }
        }
    }
}

void substitution_retroactive_row(double A_mat[N][N], double b_vec[N], double x_vec[N]){
    int col, lin;

    for(lin = N - 1; lin >= 0; lin--){
        double current_b = b_vec[lin];
        double terms_to_subtract = 0.0;
        #pragma omp parallel for reduction(+:terms_to_subtract)
        for(col = lin + 1; col < N; col++){
            terms_to_subtract += A_mat[lin][col] * x_vec[col];
        }
        x_vec[lin] = (current_b - terms_to_subtract) / A_mat[lin][lin];
    }
}

void substitution_retroactive_column(double A_mat[N][N], double b_vec[N], double x_vec[N]){
    int i, col, lin;

    #pragma omp parallel for
    for(i = 0; i < N; i++){
        x_vec[i] = b_vec[i];
    }

    for(col = N - 1; col >= 0; col--){
         x_vec[col] /= A_mat[col][col];
        for (lin = 0; lin < col; lin++)
            x_vec[lin] -= A_mat[lin][col] * x_vec[col];
    }
}

void substitution_retroactive_column_inner(double A_mat[N][N], double b_vec[N], double x_vec[N]){
    int i, col, lin;

    for(i = 0; i < N; i++){
        x_vec[i] = b_vec[i];
    }

    for(col = N - 1; col >= 0; col--){
        #pragma omp single
        x_vec[col] /= A_mat[col][col];

        #pragma omp parallel for
        for(lin = 0; lin < col; lin++){
            x_vec[lin] -= A_mat[lin][col] * x_vec[col];
        }
    }
}

void substitution_retroactive_row_serial(double A_mat[N][N], double b_vec[N], double x_vec[N]){
    int lin, col;
    for (lin = N - 1; lin >= 0; lin--) {
        x_vec[lin] = b_vec[lin];
        for (col = lin + 1; col < N; col++)
            x_vec[lin] -= A_mat[lin][col] * x_vec[col];
        x_vec[lin] /= A_mat[lin][lin];
    }
}

void substitution_retroactive_column_serial(double A_mat[N][N], double b_vec[N], double x_vec[N]){
        int i, col, lin;
    for (i = 0; i < N; i++)
        x_vec[i] = b_vec[i];
    for (col = N - 1; col >= 0; col--) {
        x_vec[col] /= A_mat[col][col];
        for (lin = 0; lin < col; lin++)
            x_vec[lin] -= A_mat[lin][col] * x_vec[col];
    }
}

int main(){
    inicialize_system(A, b, x_lin, x_col);
    double start_time, end_time;

    printf("Laço interno das linhas\n");
    memcpy(x_lin, b, N * sizeof(double));
    start_time = omp_get_wtime();
    substitution_retroactive_row(A, b, x_lin);
    end_time = omp_get_wtime();
    printf("Tempo de execução: %f segundos\n", end_time - start_time);

    printf("Primeiro laço colunas\n");
    memcpy(x_col, b, N * sizeof(double));
    start_time = omp_get_wtime();
    substitution_retroactive_column(A, b, x_col);
    end_time = omp_get_wtime();
    printf("Tempo de execução: %f segundos\n", end_time - start_time);

    printf("Laço paralelo interno das colunas\n");
    memcpy(x_col, b, N * sizeof(double));
    start_time = omp_get_wtime();
    substitution_retroactive_column_inner(A, b, x_col);
    end_time = omp_get_wtime();
    printf("Tempo de execução: %f segundos\n", end_time - start_time);

    return 0;
}