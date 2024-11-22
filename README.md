# naturalistic-ai

This is the repository to share/centralize code for Naturalistic Uses of AI.

### Setup

Using Python >=3.8:

```
pip install -r requirements.txt
```

### Load Datasets

```
# If necessary:
huggingface-cli login

python src/scripts/load_datasets.py --dataset wildchat_v1 --save data/wildchat_v1_sample.csv --sample 100
```

### Run GPT Annotations

1. Set your OPENAI API KEY:
```
echo "OPENAI_API_KEY='<your_key>'" > data/.env
```

2. Create json file, with a list of input examples like `data/test_sample.json`.

3. Run GPT annotation on the examples, using this script:
```
python src/scripts/run_gpt.py --system_prompt_id language_id --input data/static/test_sample.json --save data/outputs.jsonl
```
