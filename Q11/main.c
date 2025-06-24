/*11. [Pacheco and Malensek, 2022] Escreva um programa que possa ser usado para determi-
nar o custo de alterar a distribuição de uma estrutura de dados distribuída. Quanto tempo
leva para mudar de uma distribuição em bloco de um vetor para uma distribuição cíclica?
Quanto tempo leva a redistribuição reversa?*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

void gen_data(int *array, int size){
    for(int i = 0; i < size; i++){
        array[i] = i;
    }
}
