#!/bin/bash

$PREFIX=$HOME/programacao-paralela/CUDA/Trapezio ; wait

cd $PREFIX ; wait

make clean ; wait
make ; wait

time ./trapezio.exe 1024 1048576 0.0 3.14159
