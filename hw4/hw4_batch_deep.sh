#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=15000
#SBATCH --output=results/deep/run7/hw4_%04a_stdout.txt
#SBATCH --error=results/deep/run7/hw4_%04a_stderr.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=hw4_Run7
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/deep_learning_practice/homework/hw4
#SBATCH --array=0-4
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# 2023
cp /home/fagg/datasets/core50/cache/cache_core50_objects_10_fold* $LSCRATCH
python hw3_base.py @oscer.txt @exp_deep.txt @net_deep.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --cache $LSCRATCH 
