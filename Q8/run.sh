#!/bin/bash

PREFIX=$HOME/Documentos/openmp/Q8 ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 5 --hostfile ./hostfile ./saida.o