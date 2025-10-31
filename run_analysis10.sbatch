#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=12:00:00
#SBATCH -J analysisRUN10
#SBATCH -o log_ana10.o%j
#SBATCH -e log_ana10.e%j
#SBATCH -p shared
#SBATCH --mail-user=lukwidmer@ucsb.edu
#SBATCH --mail-type=NONE

# Manage processing environment, load compilers and applications.
module load conda texlive
eval "$(conda shell.bash hook)"
conda activate PARTIES_postprocess_anvil


PYTHONPATH=. python ./main.py \
	--output_dir="/home/x-lwidmer/Documents/PARTIES_postprocess/output/RUN10" \
	--utexas_data_dir="/home/x-lwidmer/Documents/PARTIES_postprocess/data" \
	--parties_data_dir="/anvil/scratch/x-lwidmer/RUN10" \
	-trn \
	--num_workers=8 \
	--min_file_index=226
