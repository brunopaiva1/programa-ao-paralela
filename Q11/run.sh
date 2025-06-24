#!/bin/bash

PREFIX=$HOME/Documentos/openmp/Q11 ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 4 ./saida.o 100000000