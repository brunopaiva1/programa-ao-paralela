#!/bin/bash

PREFIX=$HOME/home/bruno/programacao-paralela/MPI/Monte_carlo ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 4 ./saida.exe 100000000