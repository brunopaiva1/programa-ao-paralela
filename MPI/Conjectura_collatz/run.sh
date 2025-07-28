#!/bin/bash

PREFIX=$HOME/Documentos/openmp/Q15 ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 4 ./saida.o