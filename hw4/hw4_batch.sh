#!/bin/bash
#
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=20000
#SBATCH --output=results/shallow/run1/hw4_%04a_stdout.txt
#SBATCH --error=results/shallow/run1/hw4_%04a_stderr.txt
#SBATCH --time=00:15:00
#SBATCH --job-name=hw4_Run1
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/deep_learning_practice/homework/hw4
#SBATCH --array=0
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# 2023
cp /home/fagg/datasets/core50/cache/cache_core50_objects_10_fold* $LSCRATCH
python hw3_base.py @oscer.txt @exp.txt @net_shallow.txt --exp_index $SLURM_ARRAY_TASK_ID --cache $LSCRATCH --gpu
