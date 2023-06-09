#!/bin/bash
# Name of the job
#SBATCH -J rlprompt
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
#SBATCH --array=0-4
# Start your application
eval "$(conda shell.bash hook)"

N="$SLURM_ARRAY_TASK_ID"
conda activate rlprompt
seeds=(40 41 42 43 44)
seed=${seeds[N]}
python run_fsc.py \
    dataset=mrpc \
    dataset_seed=0 \
    prompt_length=5 \
    task_lm=google/flan-t5-base \
    random_seed=$seed