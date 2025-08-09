# !/bin/bash

$PREFIX=$HOME/programacao-paralela/CUDA/Matriz ; wait

cd $PREFIX ; wait

make clean ; wait
make ; wait

time ./matriz.exe 1024
