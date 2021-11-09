#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=20        # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --partition=cpu-long
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=results/kmeans/kmeans_%j.out  # File to which STDOUT will be written
#SBATCH --error=results/kmeans/kmeans_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=andreas.hochlehnert@uni-tuebingen.de  # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

# echo $SLURM_JOB_ID

# insert your commands here
singularity exec docker://libeanim/ml-research:base python code/counterfactual_xai/kmeans/run_kmeans.py
