## Anthropic-Clio Annotation Script

This script is based on the dataset for hierarchical task clusters and the associated paper on annotation prompts.

### Implementation Details

To optimize asynchronous use of the GPT API, hierarchical task annotation is implemented in an iterative manner. If a recursive cluster does not contain multiple sub-tasks, the script automatically proceeds to annotate the next task without invoking the LLM. Caching is handled through the `GPT` class to minimize redundant API calls.

### Usage

```bash
python -m src.clio_scripts.run_clio_with_gpt \
  --input data/test/wildchat4k-raw.json \
  --max_samples 20 \
  --model_id command-a-03-2025 \
  --api_key_env COHERE_API_KEY \
  --api_base_url https://api.cohere.ai/compatibility/v1/chat/completions \
  --api_disable_token_counting \
  --save data/test/test-clio.json
```

To switch the annotator LLM to other models, replace the `--model_id`, `--api_key_env` and `--api_base_url` arguments with the appropriate values for the other APIs.

