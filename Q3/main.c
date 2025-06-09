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

int main(){


    return 0;
}