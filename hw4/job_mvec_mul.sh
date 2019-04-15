#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=ac6361_mvec_mul
#SBATCH --output=mvec_mul_%j.out
#SBATCH --err=mvec_mul_%j.err
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END
#SBATCH --mail-user=ac6361@nyu.edu
 
srun ./mvec_mul
