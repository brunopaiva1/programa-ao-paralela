#!/bin/bash

PREFIX=$HOME/openmp/Q1 ; wait

cd $PREFIX ; wait

make -j ; wait

./main.o 4 3000