#!/bin/bash

PREFIX=$HOME/programacao-paralela/MPI/Anel ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 2 ./saida.exe 4