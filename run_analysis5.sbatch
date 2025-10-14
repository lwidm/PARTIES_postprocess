#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:00:00
#SBATCH -J analysisRUN5
#SBATCH -o log_ana5.o%j
#SBATCH -e log_ana5.e%j
#SBATCH -p shared
#SBATCH --mail-user=lukwidmer@ucsb.edu
#SBATCH --mail-type=NONE

# Manage processing environment, load compilers and applications.
module load conda texlive
eval "$(conda shell.bash hook)"
conda activate PARTIES_postprocess_anvil

PYTHONPATH=. python ./main.py \
	--output_dir="/home/x-lwidmer/Documents/PARTIES_postprocess/outputRUN5" \
	--utexas_data_dir="/home/x-lwidmer/Documents/PARTIES_postprocess/data" \
	--parties_data_dir="/anvil/scratch/x-lwidmer/RUN5" \
	--min_file_index=101
	# --max_file_index=138
