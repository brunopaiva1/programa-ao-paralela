# !/bin/bash

$PREFIX=$HOME/programacao-paralela/CUDA/Matriz ; wait

cd $PREFIX

make clean
make

time ./matriz.exe 1024
