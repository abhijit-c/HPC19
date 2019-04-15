#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=ac6361_poisson
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END
#SBATCH --mail-user=ac6361@nyu.edu
 
srun ./poisson_jacobi
