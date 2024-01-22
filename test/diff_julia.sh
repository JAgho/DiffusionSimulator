#!/bin/bash
#SBATCH --job-name=recon_jul
#SBATCH --partition=cubric-dgx
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --output=./output/slurm.out

julia test/diff_rve.jl
