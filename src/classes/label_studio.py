import os
import sys
import typing
import json
import re
from collections import defaultdict

sys.path.append("./")

# from src.classes.message import Message
import src.helpers.io as io
from src.classes.annotation_set import AnnotationSet
from src.classes.annotation_record import AnnotationRecord




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