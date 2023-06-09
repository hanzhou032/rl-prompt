#!/bin/bash
# Name of the job
#SBATCH -J evalrlprompt
# time: 48 hours
#SBATCH --time=999:0:0
# Number of GPU
#SBATCH --gres=gpu:rtx_6000_ada:1
# Number of cpus
#SBATCH --cpus-per-task=2
# Log output
#SBATCH -e ./log/slurm-err-%j.txt
#SBATCH -o ./log/slurm-out-%j.txt
#SBATCH --open-mode=append
#SBATCH --array=0
# Start your application
eval "$(conda shell.bash hook)"

N="$SLURM_ARRAY_TASK_ID"
conda activate rlprompt
seed=42
python run_eval.py \
    dataset=qnli \
    task_lm=google/flan-t5-base \
    prompt=\"\"