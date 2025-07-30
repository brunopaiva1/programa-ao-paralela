#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=24
#SBATCH -p sequana_cpu_dev
#SBATCH -J exemplo
#SBATCH --exclusive

echo $SLUM_JOB_NODELIST

cd /scratch/pex1272-ufersa/bruno.silva7/programacao-paralela/CUDA/Bitonic_sort

module load gcc/14.2.0_sequana
module load openmpi/gnu/4.1.4+gcc-12.4+cuda-11.6_sequana

make clean
make

./bitonic_sort 1024