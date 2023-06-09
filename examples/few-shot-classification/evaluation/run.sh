#!/bin/bash
python run_eval.py \
    dataset=sst2 \
    task_lm=google/flan-t5-base \
    prompt=\"ReviewCustomerBuyBookBook\"