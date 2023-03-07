#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=30000
#SBATCH --output=results/deep/run2/hw3_%04a_stdout.txt
#SBATCH --error=results/deep/run2/hw3_%04a_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=HW3_Run2 
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/deep_learning_practice/homework/hw3
#SBATCH --array=0-4
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# 2023
python hw3_base.py @oscer.txt @exp_deep.txt @net_deep.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK 
