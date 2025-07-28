#!/bin/bash

PREFIX=$HOME/Documentos/openmp/Q10 ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 4 ./saida.o 1000