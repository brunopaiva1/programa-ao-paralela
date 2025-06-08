#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 1000

void inicialize_system(double A[N][N], double b[N], double x[N]){
    int i, j;
    for(i = 0; i < N; i++){
        b[i] = (double)(i + 1);
        x[i] = 0.0;
        for(j = 0; j < N; j++){
            if(j >= i){
                A[i][j] = (double)(i + j + 1);
                if(i == j){
                    A[i][j] += 10.0;
                }
            } else {
                A[i][j] = 0.0;
            }
        }
    }
}
int main(){


    return 0;
}