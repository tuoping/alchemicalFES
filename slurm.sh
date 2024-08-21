#!/bin/bash -l
#SBATCH --parsable
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --gres=gpu:1
#SBATCH --partition gpu100
#SBATCH --mem-per-gpu=24g
#SBATCH --time=48:00:00
#SBATCH --constraint=bookworm
#SBATCH --exclude=

# module purge
# module load openmpi
# export OMP_NUM_THREADS=16
# export TF_INTRA_OP_PARALLELISM_THREADS=16
# export TF_INTER_OP_PARALLELISM_THREADS=4
# 
# source /nfs/scistore14/chenggrp/ptuo/pkgs/deepmd-kit/sourceme.sh
 
# bash work.sh
sleep 48h
