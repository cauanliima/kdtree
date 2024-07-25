#!/bin/bash
#SBATCH --partition intel-128
#SBATCH --nodes 1
#SBATCH --time 00:02:0
#SBATCH --job-name prog
#SBATCH --output prog-%j.out
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --exclusive
srun ./src/tree_omp.x ./benchmark/foo.csv 
