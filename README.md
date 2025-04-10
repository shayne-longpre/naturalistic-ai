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

2. Create json file, with a list of input examples like `data/tests/sample.json`.

3. Run GPT annotation on the examples, using this script:
```
bash run_gpt.sh
```

The `run_gpt.sh` script supports various arguments:
- `level_id`: Prompt level ID [str]
- `prompt_id`: Prompt system ID [str]
- `model_id`: Name of model (choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o3-mini"]) [str]
- `input`: Input json file [str]
- `save`: Output jsonl file [str]

The current `run_gpt.sh` has sample commands for running **1)** Media format (prompt level), **2)** Answer form (response level), and **3)** Topic (turn level) for test json file in `data/static/test_cedric.json` using GPTo3-mini.