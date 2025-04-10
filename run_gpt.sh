python -u src/scripts/run_gpt.py \
    --input_format free \
    --level_id prompt \
    --prompt_id media_format \
    --model_id o3-mini \
    --input data/static/test_cedric.json \
    --save res/tests/test_prompt_media_format.jsonl > log/tests/test_prompt_media_format.log;


python -u src/scripts/run_gpt.py \
    --input_format free \
    --level_id response \
    --prompt_id answer_form \
    --model_id o3-mini \
    --input data/static/test_cedric.json \
    --save res/tests/test_response_answer_form.jsonl > log/tests/test_response_answer_form.log;


python -u src/scripts/run_gpt.py \
    --input_format free \
    --level_id turn \
    --prompt_id topic \
    --model_id o3-mini \
    --input data/static/test_cedric.json \
    --save res/tests/test_turn_topic.jsonl > log/tests/test_turn_topic.log