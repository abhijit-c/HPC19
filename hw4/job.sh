#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=myFirstTest
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=1GB
 
srun ./MMultBLAS


