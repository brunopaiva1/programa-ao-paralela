/* 4. [Pacheco and Malensek, 2022] Use OpenMP para implementar um programa que faça
eliminação gaussiana (veja o problema anterior). Você pode assumir que o sistema de
entrada não precisa de nenhuma troca de linha. */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-4

void eliminacao_gaussiana_p(float **A, float *B, int n) {
  int pivot, col, lin;

  for (pivot = 0; pivot < n - 1; pivot++) {
    if (A[pivot][pivot] == 0.0) {
      printf("Erro: pivô nulo encontrado na linha %d.\n", pivot);
      exit(1);
    }

#pragma omp parallel for default(none) shared(A, B, n, pivot) private(lin, col)
    for (lin = pivot + 1; lin < n; lin++) {
      float fator = A[lin][pivot] / A[pivot][pivot];

#pragma omp simd
      for (col = pivot; col < n; col++) {
        A[lin][col] -= fator * A[pivot][col];
      }

      B[lin] -= fator * B[pivot];
    }
  }
}

void eliminacao_gaussiana(float **A, float *B, int n) {
  for (int pivot = 0; pivot < n - 1; pivot++) {
    if (A[pivot][pivot] == 0.0) {
      printf("Erro: pivô nulo encontrado na linha %d.\n", pivot);
      exit(1);
    }

    for (int lin = pivot + 1; lin < n; lin++) {
      float fator = A[lin][pivot] / A[pivot][pivot];

      for (int col = pivot; col < n; col++) {
        A[lin][col] -= fator * A[pivot][col];
      }

      B[lin] -= fator * B[pivot];
    }
  }
}

void validacao(float **A, float **B, int n) {
  int ok = 1;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (fabs(A[i][j] - B[i][j]) > EPSILON) {
        ok = 0;
        break;
      }
    }
  }

  printf("Resultado: %s\n", (ok == 0) ? "ERRADO" : "CORRETO");
  return;
}

int main(int argc, char *argv[]) {
  srand(27);
  int n = atoi(argv[1]);
  double start, end;

  float **A = (float **)malloc(sizeof(float *) * n);
  float **A_copy = (float **)malloc(sizeof(float *) * n);
  float *B = (float *)malloc(sizeof(float) * n);

  for (int i = 0; i < n; i++) {
    A[i] = (float *)malloc(sizeof(float) * n);
    A_copy[i] = (float *)malloc(sizeof(float) * n);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = rand() % 9 + 1;
      A_copy[i][j] = A[i][j];
    }
  }

  start = omp_get_wtime();
  eliminacao_gaussiana(A, B, n);
  end = omp_get_wtime();
  printf("Tempo Sequencial: %f\n", end - start);

  start = omp_get_wtime();
  eliminacao_gaussiana_p(A_copy, B, n);
  end = omp_get_wtime();
  printf("Tempo Paralelo: %f\n", end - start);

  validacao(A, A_copy, n);

  for (int i = 0; i < n; i++) {
    free(A[i]);
    free(A_copy[i]);
  }
  free(A);
  free(A_copy);

  free(B);

  return 0;
}