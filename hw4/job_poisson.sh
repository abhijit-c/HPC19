#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=ac6361_poisson
#SBATCH --output=poisson_%j.out
#SBATCH --err=poisson_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END
#SBATCH --mail-user=ac6361@nyu.edu
 
srun ./poisson_jacobi
