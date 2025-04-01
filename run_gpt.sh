#!/bin/bash

python -u src/scripts/run_gpt.py \
    --level_id prompt \
    --prompt_id media_format \
    --model_id o3-mini \
    --input data/static/test_cedric.json \
    --save res/test_up_mf.jsonl > log/test.log;


python -u src/scripts/run_gpt.py \
    --level_id response \
    --prompt_id answer_form \
    --model_id o3-mini \
    --input data/static/test_cedric.json \
    --save res/test_mr_af.jsonl > log/test2.log;


python -u src/scripts/run_gpt.py \
    --level_id turn \
    --prompt_id topic \
    --model_id o3-mini \
    --input data/static/test_cedric.json \
    --save res/test_t_t.jsonl > log/test3.log