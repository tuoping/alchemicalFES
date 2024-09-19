#!/bin/bash -l
#SBATCH --parsable
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --mem-per-gpu=24g
#SBATCH --time=48:00:00
#SBATCH --constraint=bookworm
#SBATCH --exclude=gpu113,gpu118,gpu119,gpu123,gpu124,gpu125,gpu127,gpu137,gpu138,gpu139,gpu145,gpu144,gpu148,gpu150

# module purge
# module load openmpi
# export OMP_NUM_THREADS=16
# export TF_INTRA_OP_PARALLELISM_THREADS=16
# export TF_INTER_OP_PARALLELISM_THREADS=4
# 
# source /nfs/scistore14/chenggrp/ptuo/pkgs/deepmd-kit/sourceme.sh
 
sleep 48h
# ~/pkgs/deepmd-kit/envs/seq/bin/python -u trainlatt6x6_simplex_batch1024_lrdecay.py $1 $2 $3 $4
