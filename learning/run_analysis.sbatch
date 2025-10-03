#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=1:00:00
#SBATCH -J analysis
#SBATCH -o log_ana.o%j
#SBATCH -e log_ana.e%j
#SBATCH -p wholenode
#SBATCH --mail-user=lukwidmer@ucsb.edu
#SBATCH --mail-type=all

# Manage processing environment, load compilers and applications.
module load python
source /home/x-lwidmer/Documents/PARTIES_postprocess/venv_anvil/bin/activate

python ./main.py
