#!/bin/bash
#SBATCH --job-name=recon_jul
#SBATCH --partition=cubric-gpu
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --output=./output/slurm.out

julia test/sim_bases.jl
