import pandas as pd
import json, sys, asyncio, argparse

sys.path.append("./")

from src.clio_scripts.utils import (
    init_clio_annotation_dict, 
    assign_answer_to_annotation,
    format_samples_per_clio_annotation,
    get_next_keys
)
from src.helpers import gpt, io
from src.classes.dataset import Dataset
from src.scripts.utils import batch_generator, read_json

def extract_samples_and_metadata(dataset, existing_samples=None, annotation_key="is_occupational_task", max_samples=None):
    
    samples, metadatas, conversation_ids, clio_annotations = [], [], [], []
    modified_samples, modified_metadatas, modified_conversation_ids, modified_clio_annotations = [], [], [], []

    # Add max_samples limit if specified
    if max_samples is not None:
        dataset.data = dataset.data[:max_samples]
    for _, data in enumerate(dataset.data):
        if data.conversation_id in existing_samples:
            sample_dict = existing_samples[data.conversation_id]["clio_annotation"]
            if annotation_key == "is_occupational_task" and sample_dict["is_occupational_task"] != "":
                print(f"Skipping existing conversation_id: {data.conversation_id} with completed annotation: {sample_dict}")
                continue
            elif annotation_key.startswith("cluster_") or annotation_key == "occupational_skills":
                if sample_dict["is_occupational_task"] == "":
                    print(f"Skipping existing conversation_id: {data.conversation_id} with incomplete 'is_occupational_task' annotation: {sample_dict}")
                    continue
                elif sample_dict["is_occupational_task"] == "No":
                    print(f"Skipping existing conversation_id: {data.conversation_id} with 'No' annotation for {annotation_key}: {sample_dict}")
                    continue
                elif sample_dict["is_occupational_task"] == "Yes":
                    if sample_dict[annotation_key] != "":
                        print(f"Skipping existing conversation_id: {data.conversation_id} with completed annotation: {existing_samples[data.conversation_id]}")
                        continue
                    else:
                        # if the len of next_keys is 1, modify the annotation and skip gpt call
                        next_keys = get_next_keys(sample_dict)
                        if len(next_keys) == 1:
                            sample_dict[annotation_key] = next_keys[0]
                            
                            modified_samples.append(data)
                            modified_metadatas.append(data.metadata)
                            modified_conversation_ids.append(data.conversation_id)
                            modified_clio_annotations.append(sample_dict)
                        else:
                            samples.append(data)
                            metadatas.append(data.metadata)
                            conversation_ids.append(data.conversation_id)
                            clio_annotations.append(sample_dict)
        else:
            samples.append(data)
            metadatas.append(data.metadata)
            conversation_ids.append(data.conversation_id)
            clio_annotations.append(init_clio_annotation_dict())
    formatted_samples, metadatas, conversation_ids, clio_annotations = format_samples_per_clio_annotation(samples, metadatas, conversation_ids, clio_annotations, annotation_key)
    return [formatted_samples, metadatas, conversation_ids, clio_annotations], [modified_samples, modified_metadatas, modified_conversation_ids, modified_clio_annotations]


async def run_gpt(args, batch_size=1, clio_annotation="is_occupational_task"):

    data = read_json(args.save)
    failed_data = read_json(args.save.replace('.json', '-failed.json'))
    dataset = Dataset.load(args.input)
    formatted_data, modified_data = extract_samples_and_metadata(dataset, data, clio_annotation, max_samples=args.max_samples)
    formatted_samples, metadata, conversation_ids, clio_annotations = formatted_data
    
    # modify the data without gpt call 
    modified_samples, modified_metadatas, modified_conversation_ids, modified_clio_annotations = modified_data
    for sample, meta, conv_id, anno in zip(modified_samples, modified_metadatas, modified_conversation_ids, modified_clio_annotations):
        data[conv_id]["clio_annotation"] = anno
    
    if not formatted_samples:
        print("All examples already exist in the save file.")
        return
    
    print(f"Formatted prompts: {len(formatted_samples)}, Metadata: {len(metadata)}")

    # print GPT arguments
    print(f"Running GPT with model: {args.model_id}, API base URL: {args.api_base_url}, API key environment variable: {args.api_key_env}, Disable token counting: {args.api_disable_token_counting}")
    gpt_instance = gpt.GPT(model=args.model_id, base_url=args.api_base_url, key_env=args.api_key_env, disable_token_counting=args.api_disable_token_counting)

    for batch, meta_batch, conv_batch, anno_batch in zip(
        batch_generator(formatted_samples, batch_size),
        batch_generator(metadata, batch_size),
        batch_generator(conversation_ids, batch_size),
        batch_generator(clio_annotations, batch_size)
    ):
        try:
            batch_responses = await gpt_instance.process_prompts_in_batches_async(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue        
        
        for sample, response, meta, conv_id, anno in zip(batch, batch_responses, meta_batch, conv_batch, anno_batch):
            if response is None:
                print(f"[Error] Skipping saving due to failed response for turn {conv_id}, ex_id: {meta['ex_id']}")
                if conv_id in data:
                    failed_data[conv_id].update({
                        f"{clio_annotation}_input": sample,
                    })
                else:
                    failed_data[conv_id] = {
                        **meta,
                        "model_id": args.model_id,
                        "conversation_id": conv_id,
                        f"{clio_annotation}_input": sample,
                        "clio_annotation": anno,
                    }
                continue
            else:
                # Assign the response to the appropriate annotation
                anno, response = assign_answer_to_annotation(anno, response, clio_annotation)
                if conv_id in data:
                    data[conv_id].update({
                        f"{clio_annotation}_input": sample,
                        "clio_annotation": anno,
                        f"{clio_annotation}_response": response, 
                    })
                else: 
                    data[conv_id] = {
                        **meta,
                        "model_id": args.model_id,
                        "conversation_id": conv_id,
                        f"{clio_annotation}_input": sample,
                        "clio_annotation": anno,
                        f"{clio_annotation}_response": response, 
                    }
    # save data dict to json file
    with open(args.save, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # save failed data dict to json file
    with open(args.save.replace('.json', '-failed.json'), 'w') as f:
        json.dump(failed_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset.")
    # max samples for dataset
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process from the dataset.")
    parser.add_argument("--model_id", type=str, required=True, default=None, choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o3-mini", "command-a-03-2025"])
    parser.add_argument("--api_key_env", type=str, required=True, default=None)
    parser.add_argument("--api_base_url", type=str, required=True, default=None)
    parser.add_argument("--api_disable_token_counting", required=False, action='store_true')
    parser.add_argument("--save", type=str, required=True, help="Save path.")
    args = parser.parse_args()

    # Step 1: Is occoupational task
    print(f"Running LLM tagging with model: {args.model_id} for clio annotation: is_occupational_task")
    asyncio.run(run_gpt(args, batch_size=1, clio_annotation="is_occupational_task"))

    # Step 2: Cluster 1
    print(f"Running LLM tagging with model: {args.model_id} for clio annotation: cluster_top")
    asyncio.run(run_gpt(args, batch_size=1, clio_annotation="cluster_top"))

    # Step 3: Cluster 2
    print(f"Running LLM tagging with model: {args.model_id} for clio annotation: cluster_medium")
    asyncio.run(run_gpt(args, batch_size=1, clio_annotation="cluster_medium"))

    # Step 4: Cluster 3
    print(f"Running LLM tagging with model: {args.model_id} for clio annotation: cluster_bottom")  
    asyncio.run(run_gpt(args, batch_size=1, clio_annotation="cluster_bottom"))

    # Step 5: Occupational skills
    print(f"Running LLM tagging with model: {args.model_id} for clio annotation: occupational_skills")  
    asyncio.run(run_gpt(args, batch_size=1, clio_annotation="occupational_skills"))