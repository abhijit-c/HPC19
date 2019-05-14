#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=ac6361_ssort
#SBATCH --output=ssort_%j.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --mail-user=ac6361@nyu.edu
 
srun mpirun ssort 10000
