#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --mem=10G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/home/bethge/ahochlehnert48/results/resnet50new/logs/resnet50new_%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/bethge/ahochlehnert48/results/resnet50new/logs/resnet50new_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=andreas.hochlehnert@uni-tuebingen.de  # Email to which notifications will be sent

SOURCE_DIR="/home/bethge/ahochlehnert48/code/counterfactual_xai"
# cd $(dirname "${BASH_SOURCE[0]}")
# echo `pwd`

# print info about current job
scontrol show job $SLURM_JOB_ID 

# echo $SLURM_JOB_ID

# insert your commands here
# singularity exec docker://libeanim/ml-research:base ipython Resnet50New.py -- config.json
singularity exec --nv -B /mnt/qb/datasets docker://tensorflow/tensorflow:2.2.0-gpu-jupyter ipython $SOURCE_DIR/Resnet50New.py -- $SOURCE_DIR/config.json
# srun --pty --gres=gpu:1 --mem=10G --partition=gpu-2080ti singularity exec --nv -B /mnt/qb/datasets docker://tensorflow/tensorflow:2.2.0-gpu-jupyter ipython ~/code/counterfactual_xai/Resnet50New.py -- --input