import os
import sys
import typing
import json
import re
from collections import defaultdict
import copy

sys.path.append("./")

# from src.classes.message import Message
import src.helpers.io as io
from src.classes.annotation_set import AnnotationSet
from src.classes.annotation_record import AnnotationRecord



prompt_mappings = {
    'multi_turn_relationship': 'prompt_multi_turn_relationship',
    'media_format_prompt': 'prompt_media_format',
    'interaction_features': 'prompt_interaction_features',
    'function_purpose': 'prompt_function_purpose',
    'topic_turn': 'turn_topic',
    'restricted_flags_prompt': 'turn_sensitive_use_flags',
    "other_feedback_prompt": "other_feedback_prompt",
}

response_mappings = {
    'answer_form_response': 'response_answer_form',
    'media_format_response': 'response_media_format',
    'interaction_features_response': 'response_interaction_features',
    'restricted_flags_response': 'turn_sensitive_use_flags',
    "other_feedback_response": "other_feedback_response",
}

def get_base_task_name(from_name):
    # Define mapping dictionaries for different prefixes

    # Check prompt mappings
    for old_prefix, new_prefix in prompt_mappings.items():
        if from_name == old_prefix:
            return new_prefix
    
    # Check response mappings
    for old_prefix, new_prefix in response_mappings.items():
        if from_name == old_prefix:
            return new_prefix
    
    # Handle special cases with "whole" or other patterns
    if 'whole' in from_name:
        if 'topic_turn_whole' in from_name:
            return 'turn_topic'
        elif 'other_feedback_whole' in from_name:
            return 'other_feedback'
    
    print(f"NOT FOUND: {from_name}")
    # If no mapping found, return the original without the number suffix
    # This is a fallback case
    return re.sub(r'_\d+$', '', from_name)

def load_labelstudio_v2(
    folder_path: str, 
    source: str,
    dataset_id: str,
    level: str = "conversation",
):
    """
    Load and process Label Studio annotations into annotation sets in one step.
    Works with the updated format where each file contains an array of records
    with annotations directly on each record.
    
    Args:
        folder_path: Path to the folder containing annotation JSON files
        source: Source identifier for the annotation sets
        dataset_id: Dataset identifier
        level: Level of annotations ('conversation' or 'message')
        
    Returns:
        Dictionary of AnnotationSet objects by task name
    """
    acceptable_fields = list(prompt_mappings.values()) + list(response_mappings.values())
    # Define field categories
    # prompt_fields = [
    #     "media_format_prompt", "function_purpose", "multi_turn_relationship", "interaction_features",
    #     "restricted_flags_prompt", 

    #     # "multi_turn_relationship", "media_format", "topic", 
    #     # "function_purpose", "anthropomorphization", "restricted_flags",
    #     # "interaction_features"  # Added based on the example
    # ]
    # response_fields = [
    #     "media_format_response", "answer_form_response", "interaction_features_response",
    #     "restricted_flags_response", "topic_turn_whole"

    #     # "answer_form", "self_disclosure", "topic_response", 
    #     # "media_format_response", "restricted_flags_response",
    #     # "interaction_features_response"  # Added based on the example
    # ]
    
    # Initialize task groups dictionary
    task_groups = defaultdict(list)
    from_names = set()
    
    # Process all files in the folder
    for filepath in io.listdir_nohidden(folder_path):
        if not filepath.endswith('.json'):
            continue
            
        
        # try:
        # Load JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        # Process each record in the file
        for record in records:
            record_id = record.get('id', '')
            conversation_data = record.get('data', {})
            conversation_id = conversation_data.get('conversation_id', record_id)
            
            # Extract conversations from data
            conversations = conversation_data.get('conversation', [])
            
            # Also check turn1_dialogue if it exists
            # if not conversations and 'turn1_dialogue' in conversation_data:
            #     conversations = conversation_data.get('turn1_dialogue', [])
            
            # Process annotations
            for annotation in record.get('annotations', []):
                annotator_name = annotation.get('annotator', 'Unknown')
                
                # Process results
                results = annotation.get('result', [])
                
                # Group results by turn index and task name
                results_by_turn = defaultdict(dict)
                
                for result in results:
                    from_name = result.get('from_name', '')
                    
                    # Extract turn number (e.g., "media_format_prompt_1" -> turn 1)
                    match = re.search(r'_(\d+)$', from_name)
                    if not match:
                        continue
                        
                    turn_index = int(match.group(1)) - 1
                    
                    from_names.add(from_name)
                    # Get base task name by removing turn suffix
                    # base_task_name = re.sub(r'_(prompt|response)_\d+$', '', from_name)
                    parsed_from_name = "_".join(from_name.split("_")[:-1])
                    base_task_name = get_base_task_name(parsed_from_name)
                    
                    # Skip fields not in our defined lists
                    if base_task_name not in acceptable_fields:
                        # print(base_task_name)
                        continue
                    
                    message_index = (2 * turn_index)
                    # For response fields, increment turn index
                    if "response" in base_task_name:
                        message_index += 1

                    # Store the choices
                    choices = result.get('value', {}).get('choices', [])
                    
                    # Handle special case for text fields (like other_feedback)
                    if 'text' in result.get('value', {}):
                        text_value = result.get('value', {}).get('text', [])
                        choices = text_value if isinstance(text_value, list) else [text_value]
                    
                    results_by_turn[message_index][base_task_name] = choices
                
                # Process each conversation turn with its annotations
                for turn in conversations:
                    turn_index = turn.get('turn', 0)
                    turn_annotations = results_by_turn.get(turn_index, {})
                    
                    # Use conversation_id from turn if available
                    turn_conv_id = turn.get('conversation_id', conversation_id)
                    # if conversation_id == "wildchat_20847df802a3268754fe7d7a6ada334b":
                    #     print(turn_index)
                    #     print(turn_annotations)
                    # Add annotations to task groups
                    for task_name, task_values in turn_annotations.items():
                        task_groups[task_name].append({
                            "annotation_value": task_values,
                            "conversation_id": turn_conv_id,
                            "turn_idx": turn_index,
                            "annotator_name": annotator_name
                        })

            # if conversation_id == "wildchat_20847df802a3268754fe7d7a6ada334b":
            #     print("***sdflkdsflhjsdf***----------***")
            #     print(results_by_turn)
        
        # except Exception as e:
        #     print(f"Error processing file {filename}: {e}")
    # print(from_names)
    # Convert task groups to annotation sets
    annotation_sets = {
        task_name: AnnotationSet(
            source=source,
            name=task_name,
            level=level,
            dataset_id=dataset_id,
            annotations=[
                AnnotationRecord(
                    value=x["annotation_value"],
                    target_id=f"{x['conversation_id']}-{x['turn_idx']}" if level == "message" else x['conversation_id'],
                    annotator=x.get("annotator_name")
                ) for x in data
            ]
        ) for task_name, data in task_groups.items()
    }
    
    return annotation_sets




def load_labelstudio(
    folder_path: str, 
    source: str,
    dataset_id: str,
    level: str = "conversation",
) -> dict:
    """
    Load and process Label Studio annotations into annotation sets in one step.
    
    Args:
        folder_path: Path to the folder containing annotation JSON files
        source: Source identifier for the annotation sets
        
    Returns:
        Dictionary of AnnotationSet objects by task name
    """
    # Define field categories
    prompt_fields = [
        "multi_turn_relationship", "media_format", "topic", 
        "function_purpose", "anthropomorphization", "restricted_flags"
    ]
    response_fields = [
        "answer_form", "self_disclosure", "topic_response", 
        "media_format_response", "restricted_flags_response"
    ]
    
    # Initialize task groups dictionary
    task_groups = defaultdict(list)
    
    # Process all files in the folder
    for filepath in io.listdir_nohidden(folder_path):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        annotator_name = data.get('metadata', {}).get('annotator', 'Unknown')
        
        for annotation_task in data.get('annotations', []):
            conversations = annotation_task.get('data', {}).get('conversation', [])
            
            # Collect all results
            results = []
            for annotation in annotation_task.get('annotations', []):
                results.extend(annotation.get('result', []))
            
            # Group results by turn index
            results_by_turn = defaultdict(dict)
            for result in results:
                from_name = result.get('from_name', '')
                match = re.search(r'_(\d+)$', from_name)
                if not match:
                    continue
                    
                turn_index = int(match.group(1)) - 1
                base_task_name = re.sub(r'_\d+$', '', from_name)
                
                if base_task_name not in prompt_fields + response_fields:
                    continue
                    
                if base_task_name in response_fields:
                    turn_index += 1
                
                results_by_turn[turn_index].update({
                    base_task_name: result.get('value', {}).get('choices', [])
                })
            
            # Process each conversation turn
            for turn in conversations:
                turn_index = turn.get('turn', 0)
                turn_annotations = results_by_turn.get(turn_index, {})
                
                conv_id = turn.get('conversation_id', '')
                
                # Add annotations to task groups
                for task_name, task_values in turn_annotations.items():
                    task_groups[task_name].append({
                        "annotation_value": task_values,
                        "conversation_id": conv_id,
                        "turn_idx": turn_index,
                        "annotator_name": annotator_name
                    })
    
    # Convert task groups to annotation sets
    annotation_sets = {
        task_name: AnnotationSet(
            source=source,
            name=task_name,
            level=level,
            dataset_id=dataset_id,
            annotations=[
                AnnotationRecord(
                    value=x["annotation_value"],
                    target_id=f"{x['conversation_id']}-{x['turn_idx']}" if level == "message" else x['conversation_id'],
                    annotator=x.get("annotator_name")
                ) for x in data
            ]
        ) for task_name, data in task_groups.items()
    }
    
    return annotation_sets


def split_labelstudio_files_by_conversation_id(
    input_folder: str,
    output_folder1: str,
    output_folder2: str
):
    """
    Split LabelStudio files into two folders based on conversation_id.
    For each input file, creates two output files (one in each folder)
    with annotations split by whether their conversation_ids are unique or duplicates.
    
    Args:
        input_folder: Path to folder containing LabelStudio JSON files with conversation_ids
        output_folder1: Path to folder to save files with unique conversation_ids
        output_folder2: Path to folder to save files with duplicate conversation_ids
        
    Returns:
        tuple: (count of unique conversation_ids, count of duplicate conversation_ids)
    """
    # Initialize sets to track conversation IDs
    seen_conversation_ids: Set[str] = set()
    
    # Create output folders if they don't exist
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    
    # Stats for reporting
    unique_count = 0
    duplicate_count = 0
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(input_folder, filename)
        
        try:
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create two versions of this file
            # Keep all metadata but split annotations
            output_data1 = {key: value for key, value in data.items() if key != 'annotations'}
            output_data2 = {key: value for key, value in data.items() if key != 'annotations'}
            
            # Initialize empty annotations lists
            output_data1['annotations'] = []
            output_data2['annotations'] = []
            
            # Process each annotation task
            for task in data.get('annotations', []):
                # Extract conversation data
                task_data = task.get('data', {})
                conversations = task_data.get('conversation', [])
                
                # Find all conversation_ids in this task
                conversation_ids = set()
                for turn in conversations:
                    if 'conversation_id' in turn:
                        conversation_ids.add(turn['conversation_id'])
                
                # If no conversation_ids found, default to unique (output_folder1)
                if not conversation_ids:
                    output_data1['annotations'].append(task)
                    continue
                
                # Check if any conversation_id has been seen before
                is_duplicate = any(conv_id in seen_conversation_ids for conv_id in conversation_ids)
                
                # Add to appropriate output data
                if is_duplicate:
                    output_data2['annotations'].append(task)
                    duplicate_count += 1
                else:
                    output_data1['annotations'].append(task)
                    unique_count += 1
                    # Update seen conversation IDs
                    seen_conversation_ids.update(conversation_ids)
            
            # Save output files if they have annotations
            output_path1 = os.path.join(output_folder1, filename)
            if output_data1['annotations']:
                with open(output_path1, 'w', encoding='utf-8') as f:
                    json.dump(output_data1, f, ensure_ascii=False, indent=2)
            
            output_path2 = os.path.join(output_folder2, filename)
            if output_data2['annotations']:
                with open(output_path2, 'w', encoding='utf-8') as f:
                    json.dump(output_data2, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    print(f"Split annotations into two folders:")
    print(f"  - {output_folder1}: Contains {unique_count} annotations with unique conversation IDs")
    print(f"  - {output_folder2}: Contains {duplicate_count} annotations with duplicate conversation IDs")
    
    return unique_count, duplicate_count


def split_labelstudio_files_by_conversation_id_v2(
    input_folder: str,
    output_folder1: str,
    output_folder2: str
):
    """
    Split LabelStudio files into two folders based on conversation_id.
    For each input file, creates two output files (one in each folder)
    with records split by whether their conversation_ids are unique or duplicates.
    
    Works with the updated format where each file contains an array of records
    with conversation_id inside data.conversation_id.
    
    Args:
        input_folder: Path to folder containing LabelStudio JSON files with conversation_ids
        output_folder1: Path to folder to save files with unique conversation_ids
        output_folder2: Path to folder to save files with duplicate conversation_ids
        
    Returns:
        tuple: (count of unique conversation_ids, count of duplicate conversation_ids)
    """
    # Initialize sets to track conversation IDs
    seen_conversation_ids: Set[str] = set()
    
    # Create output folders if they don't exist
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    
    # Stats for reporting
    unique_count = 0
    duplicate_count = 0
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(input_folder, filename)
        
        # try:
        # Load JSON file - in the new format, this is an array of records
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create two lists to store the split records
        unique_records = []
        duplicate_records = []
        
        # Process each record in the file
        for record in data:
            # Get the conversation_id from the record
            conversation_id = record.get('data', {}).get('conversation_id')
            
            # If no conversation_id found, check in turns
            if not conversation_id:
                conversation_turns = record.get('data', {}).get('conversation', [])
                if conversation_turns and 'conversation_id' in conversation_turns[0]:
                    conversation_id = conversation_turns[0]['conversation_id']
            
            # If still no conversation_id, default to unique (output_folder1)
            # if not conversation_id:
            #     unique_records.append(record)
            #     continue
            
            record1 = copy.deepcopy(record)
            record2 = copy.deepcopy(record)
            if len(record["annotations"]) > 1:
                record1["annotations"] = [record1["annotations"][0]]
                record2["annotations"] = [record2["annotations"][1]]

            unique_records.append(record1)
            if len(record["annotations"]) > 1:
                duplicate_records.append(record2)
                duplicate_count += 1
            else:
                unique_count += 1

            # # Check if this conversation_id has been seen before
            # if conversation_id in seen_conversation_ids:
            #     duplicate_records.append(record1)
            #     duplicate_count += 1
            # else:
            #     unique_records.append(record)
            #     unique_count += 1
            #     # Add to seen set
            #     seen_conversation_ids.add(conversation_id)
        
        # Save output files if they have records
        if unique_records:
            output_path1 = os.path.join(output_folder1, filename)
            with open(output_path1, 'w', encoding='utf-8') as f:
                json.dump(unique_records, f, ensure_ascii=False, indent=2)
        
        if duplicate_records:
            output_path2 = os.path.join(output_folder2, filename)
            with open(output_path2, 'w', encoding='utf-8') as f:
                json.dump(duplicate_records, f, ensure_ascii=False, indent=2)
                
        # except Exception as e:
        #     print(f"Error processing file {filename}: {e}")
    
    print(f"Split records into two folders:")
    print(f"  - {output_folder1}: Contains {unique_count} records with unique conversation IDs")
    print(f"  - {output_folder2}: Contains {duplicate_count} records with duplicate conversation IDs")
    
    return unique_count, duplicate_count