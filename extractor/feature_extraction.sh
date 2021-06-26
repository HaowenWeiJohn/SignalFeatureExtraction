#!/bin/bash
#SBATCH --job-name="feature_extraction"
#SBATCH -D .
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=350G
#SBATCH --cpus-per-task=5

#SBATCH --partition=long
#SBATCH --time=30:00:00


module purge
module load gcc/6.3.0 slurm/17.02.11 cudnn/8.1.1.33-11.2/3k5bbs63 cuda/11.0.2/ur4svoci 


python feature_extraction.py