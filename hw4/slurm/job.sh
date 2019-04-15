#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=ac6361_poisson
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
 
srun ./poisson_jacobi
