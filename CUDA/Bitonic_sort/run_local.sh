#!/bin/bash

$PREFIX=$HOME/programacao-paralela/CUDA/Bitonic_sort ; wait

cd $PREFIX ; wait

make clean ; wait
make ; wait

time ./bitonic_sort.exe 16
