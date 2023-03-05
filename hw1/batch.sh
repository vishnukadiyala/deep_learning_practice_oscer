#!/bin/bash

# Andrew H. Fagg
#
# Example with one experiment
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments!)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu
#

#
#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %j is translated into the job number
#SBATCH --output=results/r1/hw1_%j_stdout.txt
#SBATCH --error=results/r1/hw1_%j_stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=hw1_run4
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/deep_learning_practice/homework/hw1
#SBATCH --array=0-160
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# Change this line to start an instance of your experiment
python hw1_base.py --hidden 1000 --activation_out linear --epochs 1000 --results_path ./results/r1 --exp_index $SLURM_ARRAY_TASK_ID --output_type ddtheta --predict_dim 0

