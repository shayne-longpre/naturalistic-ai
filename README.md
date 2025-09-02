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
- `input_format`: Type of prompt used for passing conversation logs ("free": as free-text, "json": in distinct JSON format)
- `level_id`: Prompt level ID [str]
- `prompt_id`: Prompt system ID [str]
- `model_id`: Name of model (choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o3-mini"]) [str]
- `input`: Input json file [str]
- `save`: Output jsonl file [str]

The current `run_gpt.sh` has sample commands for running **1)** Media format (prompt level), **2)** Answer form (response level), and **3)** Topic (turn level) for test json file in `data/static/test_cedric.json` using GPTo3-mini.

### Run Simple Annotations 

The simple annotations include annotations for the text length, word count, character count, token count, and language detection via [Lingua](https://github.com/pemistahl/lingua-py). Running the simple annotation pipeline requires 2 steps: 

1. Find the path to a json file of your examples to annotate, as with the GPT annotations above. 

2. Run the script with the following arguments:

```
python run_simple_annotations.py \
  --input path/to/input.json \
  --save path/to/output/annotations_folder      <-this is a folder, not a file! A file will be created for each feature. 
```

Each output file will be a json file with the same IDs as your input file. For the language annotation, the annotation will be a list of all languages found in the text through the Lingua library. For specific details / choices on language identification, see the documentation for the language prediction function [here](https://github.com/shayne-longpre/naturalistic-ai/blob/d2150b3b07946c8e826a45e46939e6af829d57e4/src/scripts/run_simple_annotations.py#L54). For all other annotations, the annotation will be a string of the annotation measure. 

### Run Evaluations

1. Run evaluation on the GPT annotations, using this script:
```
python -u src/scripts/evaluator/form_checker.py \
    --input_dir $INPUT_DIRECTORY \
    --save $SAVE_CSV_FILE;
```

- `input_dir`: Input directory containing GPT annotations (in jsonl files) to evaluate
- `save`: Name of csv file to save the evaluation results


2. Output will be saved as csv file in the following format:

```csv
level_id,prompt_id,total_entries,invalid_Invalid JSON list,invalid_Item is not a dictionary,invalid_Missing keys,invalid_Confidence out of range,invalid_Invalid option,conf_[0.0-0.2),conf_[0.2-0.4),conf_[0.4-0.6),conf_[0.6-0.8),conf_[0.8-1.0],avg_preds_per_row,unique_labels,label_entropy,label_gini
prompt,interaction_features,597,0,0,1,0,0,0.96,1.11,0.32,2.71,94.9,1.05,6 / 6,1.0856,0.3693
prompt,media_format,597,1,0,2,0,4,0.0,0.09,0.27,1.01,98.63,1.85,11 / 11,1.9362,0.651
prompt,multi_turn_relationship,597,0,0,0,0,0,0.0,0.33,0.16,2.14,97.36,1.02,5 / 5,2.2077,0.7654
prompt,topic,597,0,0,1,0,12,0.13,0.13,1.14,14.16,84.45,1.35,36 / 39,4.4549,0.9395
response,answer_form,597,0,0,0,0,0,0.0,0.16,0.0,1.63,98.21,1.03,5 / 6,0.8211,0.247
response,interaction_features,597,0,0,0,0,0,0.2,0.2,0.2,2.85,96.54,0.82,8 / 9,1.3246,0.4023
response,media_format,597,0,0,0,0,1,0.0,0.0,0.08,1.45,98.47,2.08,10 / 11,1.942,0.6761
turn,sensitive_use_flags,597,0,0,0,0,1,0.0,0.0,0.0,4.11,95.89,0.9,19 / 24,1.0779,0.2639
turn,topic,597,0,0,0,0,5,0.17,0.09,1.14,27.3,71.3,1.93,35 / 37,3.9428,0.8918
```

- `level_id`: Prompt level ID
- `prompt_id`: Prompt system ID
- `total_entries`: Total number of entries used in evaluation (597 for 120 rows)
- Errors are categorized in the following groups:
   - `Invalid JSON`: Model response not parseable as JSON
   - `Invalid dict`: Parseable but not in dictionary format
   - `Missing keys`: Parseable but "labels" and "confidence" keys are missing
   - `Confidence range`: Parseable but confidence score is out of range (not in 0-1)
   - `Invalid option`: Parseable but the option generated is not in the taxonomy's options
- `conf_{bucket}`: Confidence buckets (from 0.0 to 1.0)
- `avg_preds_per_row`: Average prediction (labels) per annotation (for multi-label)
- `unique_labels`: Number of labels used / Total number of labels
- `label_entropy`, `label_gini`: Measure of diversity of label predictions across the responses (higher, more diverse)
