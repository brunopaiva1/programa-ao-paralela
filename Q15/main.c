/*15. [Silva et al., 2022] A Conjectura de Collatz foi desenvolvida pelo matemático alemão
Lothar Collatz. Nela, escolhendo-se um número natural inicial N , onde N > 0, os se-
guintes critérios serão obedecidos: Se N for par o seu sucessor será a metade e se N
for ímpar o seu sucessor será o triplo mais um, gerando então um novo número. Esse
processo repete-se até que eventualmente se atinja o número 1.
Use o modelo “saco de tarefa” para resolver esse problema em paralelo e verificar se
os números inteiros no intervalo de 1 a 100.000.000 atendem a esses critérios e o maior
número de passos necessários para chegar até o número 1. Atribua a um dos processos
o papel de “mestre” (gerencia a distribuição de tarefas) e aos demais processos o papel
de “trabalhadores” (executam a tarefa de avaliar se um número está dentro da conjec-
tura). Utilize mensagens de envio e recepção não bloqueantes de forma a paralelizar a
computação com o envio de mensagens.*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int collatz_steps(unsigned int n){
    int steps = 0; 
    while(n != 1){
        if(n % 2 == 0) {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        steps++;
    }
    return steps;
}

int main(int argc, char *argv[]){
return 0;
}