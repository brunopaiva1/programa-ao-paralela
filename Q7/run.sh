#!/bin/bash

PREFIX=$HOME/openmp/Q7 ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 4 ./saida 100000000