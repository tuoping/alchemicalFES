#!/bin/bash -l
#SBATCH --parsable
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --mem-per-gpu=24G
#SBATCH --time=3:00:00
#SBATCH --exclude=gpu113,gpu114,gpu118,gpu119,gpu123,gpu124,gpu125,gpu126,gpu127,gpu136,gpu137,gpu138,gpu139,gpu144,gpu145,gpu146,gpu147,gpu148,gpu150

# source /nfs/scistore14/chenggrp/ptuo/pkgs/deepmd-kit/sourceme.sh
# conda activate seq


~/pkgs/deepmd-kit/envs/seq/bin/python trainlatt6x6_simplex_batch1024_ft15.py $1 $2
