# Prompt level
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id prompt \
    --prompt_id media_format \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_p_mf \
    --input data/wildchat_small10.json \
    --save res/wildchat/prompt_media_format.jsonl > log/wildchat/prompt_media_format.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id prompt \
    --prompt_id interaction_features \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_p_if \
    --input data/wildchat_small10.json \
    --save res/wildchat/prompt_interaction_features.jsonl > log/wildchat/prompt_interaction_features.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id prompt \
    --prompt_id function_purpose \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_p_fp \
    --input data/wildchat_small10.json \
    --save res/wildchat/prompt_function_purpose.jsonl > log/wildchat/prompt_function_purpose.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id prompt \
    --prompt_id multi_turn_relationship \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_p_mtr \
    --input data/wildchat_small10.json \
    --save res/wildchat/prompt_multi_turn_relationship.jsonl > log/wildchat/prompt_multi_turn_relationship.log;


# Response level
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id response \
    --prompt_id media_format \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_r_mf \
    --input data/wildchat_small10.json \
    --save res/wildchat/response_media_format.jsonl > log/wildchat/response_media_format.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id response \
    --prompt_id interaction_features \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_r_if \
    --input data/wildchat_small10.json \
    --save res/wildchat/response_interaction_features.jsonl > log/wildchat/response_interaction_features.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id response \
    --prompt_id answer_form \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_r_af \
    --input data/wildchat_small10.json \
    --save res/wildchat/response_answer_form.jsonl > log/wildchat/response_answer_form.log;


# Turn level
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id turn \
    --prompt_id topic \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_t_t \
    --input data/wildchat_small10.json \
    --save res/wildchat/turn_topic.jsonl > log/wildchat/turn_topic.log;
python -u src/scripts/run_gpt.py \
    --input_format json \
    --level_id turn \
    --prompt_id sensitive_use_flags \
    --model_id gpt-4.1 \
    --multi_hist \
    --version wildchat_t_suf \
    --input data/wildchat_small10.json \
    --save res/wildchat/turn_sensitive_use_flags.jsonl > log/wildchat/turn_sensitive_use_flags.log;
