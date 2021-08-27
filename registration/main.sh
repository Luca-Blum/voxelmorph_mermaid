#!/bin/bash
#SBATCH --output=log/%j.out
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
source /itet-stor/lblum/net_scratch/conda/etc/profile.d/conda.sh
conda activate seminar
python -u main.py "$@"
