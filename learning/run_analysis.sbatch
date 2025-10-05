#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-tas=64
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:00:00
#SBATCH -J analysis
#SBATCH -o log_ana.o%j
#SBATCH -e log_ana.e%j
#SBATCH -p shared
#SBATCH --mail-user=lukwidmer@ucsb.edu
#SBATCH --mail-type=NONE

# Manage processing environment, load compilers and applications.
module load conda texlive
eval "$(conda shell.bash hook)"
conda activate PARTIES_postprocess_anvil

python ./main.py
