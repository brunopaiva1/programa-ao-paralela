#!/bin/bash

PREFIX=$HOME/openmp/Q5 ; wait

cd $PREFIX ; wait

make -j ; wait

./main.o texto1.txt texto2.txt texto3.txt ; wait