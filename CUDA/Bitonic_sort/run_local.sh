#!/bin/bash

$PREFIX=$HOME/programacao-paralela/CUDA/Bitonic_sort ; wait

cd $PREFIX

make clean
make

time ./bitonic_sort.exe 16
