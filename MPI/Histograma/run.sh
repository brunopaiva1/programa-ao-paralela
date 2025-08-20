#!/bin/bash

PREFIX=$HOME/programacao-paralela/MPI/Histograma ; wait

cd $PREFIX ; wait

make -j ; wait

mpiexec -np 4 ./saida.exe > histograma.txt

python3 main.py
