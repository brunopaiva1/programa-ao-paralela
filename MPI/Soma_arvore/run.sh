#!/bin/bash

PREFIX=$HOME/Documentos/openmp/Q8 ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 6 --hostfile ./hostfile ./saida.o