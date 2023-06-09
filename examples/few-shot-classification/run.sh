#!/bin/bash
python run_fsc.py \
    dataset=sst2 \
    dataset_seed=0 \
    prompt_length=5 \
    task_lm=google/flan-t5-base \
    random_seed=40