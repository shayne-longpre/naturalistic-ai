#!bin/bash

python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id prompt \
    --prompt_id interaction_features \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/prompt_interaction_features.jsonl \
    --multi_hist > log/prompt_interaction_features.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id prompt \
    --prompt_id media_format \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/prompt_media_format.jsonl \
    --multi_hist > log/prompt_media_format.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id prompt \
    --prompt_id function_purpose \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/prompt_function_purpose.jsonl \
    --multi_hist > log/prompt_function_purpose.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id prompt \
    --prompt_id multi_turn_relationship \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/prompt_multi_turn_relationship.jsonl \
    --multi_hist > log/prompt_multi_turn_relationship.log;

python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id response \
    --prompt_id media_format \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/response_media_format.jsonl \
    --multi_hist > log/response_media_format.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id response \
    --prompt_id interaction_features \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/response_interaction_features.jsonl \
    --multi_hist > log/response_interaction_features.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id response \
    --prompt_id answer_form \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/response_answer_form.jsonl \
    --multi_hist > log/response_answer_form.log;

python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id turn \
    --prompt_id topic \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/turn_topic.jsonl \
    --multi_hist > log/turn_topic.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id turn \
    --prompt_id sensitive_use_flags \
    --model_id gpt-4.1 \
    --input data/sample120.json \
    --version json_conv_multi \
    --save res/turn_sensitive_use_flags.jsonl \
    --multi_hist > log/turn_sensitive_use_flags.log;
