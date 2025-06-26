#!/bin/bash

PREFIX=$HOME/Documentos/openmp/Q13 ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 2 ./saida.o 2