#!/bin/bash
#SBATCH --partition intel-128
#SBATCH --job-name=PaScal_job
#SBATCH --output=PaScal_job%j.out
#SBATCH --error=PaScal_job%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-0:90
#SBATCH --exclusive
pascalanalyzer  -t aut -c 1 -i ../../benchmark/5M,../../benchmark/10M,../../benchmark/20M,../../benchmark/40M,../../benchmark/80M,../../benchmark/160M -r 1 -o out.json ./tree_serial.x
