#!/bin/bash
#SBATCH --job-name="featureExtract"
#SBATCH -D .
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --exclude=compute-0-28

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:3
#SBATCH --partition=long
#SBATCH --time=90:00:00


module purge
module load gcc/6.3.0 slurm/17.02.11 matlab/R2020a


matlab -nojvm -nodisplay -nosplash < regrassion_pact8.m
