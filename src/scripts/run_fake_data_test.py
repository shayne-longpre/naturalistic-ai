import os 
import sys
import pandas as pd 
import json
from typing import List, Dict, Any, Set

sys.path.append("./")

from src.helpers.io import read_jsonl, write_json, listdir_nohidden
from src.classes.dataset import Dataset
from src.classes.label_studio import load_labelstudio, load_labelstudio_v2, split_labelstudio_files_by_conversation_id, split_labelstudio_files_by_conversation_id_v2
from src.classes.annotation_set import AnnotationSet
from src.helpers.visualisation import tabulate_annotation_pair_summary, barplot_distribution, plot_confusion_matrix



def add_conversation_ids_to_label_studio_files_v2(
    label_studio_folder: str, 
    conversations_file_path: str, 
    output_folder: str = None
):
    """
    Add conversation_id to Label Studio JSON files by matching text content from a conversations file.
    Works with the new format where each record has a conversation array inside the data property.
    
    Args:
        label_studio_folder: Path to folder containing Label Studio JSON files
        conversations_file_path: Path to the file containing conversations with IDs
        output_folder: Path to save the modified files (if None, will modify in place)
        
    Returns:
        None (files are modified in place or saved to output_folder)
    """
    # Create a dictionary for fast text lookup
    text_to_conversation_id = {}
    
    # Load conversations file
    print(f"Loading conversations from {conversations_file_path}")
    with open(conversations_file_path, 'r', encoding='utf-8') as f:
        conversation_data = json.load(f)["data"]
        for conversation_datum in conversation_data:
            conversation_id = conversation_datum.get('conversation_id')
            
            # Skip if no conversation_id is found
            if not conversation_id:
                continue
            
            # Add each turn's text to the lookup dictionary
            conv_text = ""
            for turn in conversation_datum.get('conversation', []):
                # In the conversations file, content is under 'content'
                conv_text += turn.get('content', '')

            if conv_text in text_to_conversation_id:
                print(f"Warning: Duplicate text found for conversation ID: {conversation_id}")
            if conv_text:
                # Use the text as key to find the conversation_id later
                text_to_conversation_id[conv_text] = conversation_id
    
    print(f"Loaded {len(text_to_conversation_id)} conversations.")
    
    # If output_folder is provided but doesn't exist, create it
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each Label Studio file
    file_count = 0
    
    # Process a single file directly if it ends with .json
    if label_studio_folder.endswith('.json'):
        files_to_process = [label_studio_folder]
        label_studio_folder = os.path.dirname(label_studio_folder)
    else:
        files_to_process = [os.path.join(label_studio_folder, filename) 
                           for filename in os.listdir(label_studio_folder) 
                           if filename.endswith('.json')]
    
    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        
        # try:
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        
        # New format: data is a list of records
        for record in data:
            # Get conversation from the data property
            conversation_obj = record.get('data', {})
            
            # Check different places where conversation might be stored
            conversation_turns = conversation_obj.get('conversation', [])
            turn1_dialogue = conversation_obj.get('turn1_dialogue', [])
            
            # Reconstruct the complete conversation text to match with our dictionary
            conv_text = ""
            for turn in conversation_turns:
                conv_text += turn.get('content', '')
            
            # If conv_text is empty, try with turn1_dialogue
            if not conv_text and turn1_dialogue:
                for turn in turn1_dialogue:
                    conv_text += turn.get('text', '')
            
            # Try to find a matching conversation_id
            conversation_id = text_to_conversation_id.get(conv_text)
            
            if conversation_id:
                # Set the conversation_id in the record
                record['id'] = f"conv_{conversation_id.split('_')[-1]}" if conversation_id.startswith('conv_') else conversation_id
                conversation_obj['conversation_id'] = conversation_id
                
                # Also update the conversation_id in each turn
                for turn in conversation_turns:
                    turn['conversation_id'] = conversation_id
                
                for turn in turn1_dialogue:
                    turn['conversation_id'] = conversation_id
                
                modified = True
        
        # Save the modified file
        if modified:
            output_path = os.path.join(output_folder, filename) if output_folder else file_path
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            file_count += 1
            print(f"Updated file: {filename}")
                
        # except Exception as e:
        #     print(f"Error processing file {filename}: {e}")
    
    print(f"Added conversation IDs to {file_count} files")

def add_conversation_ids_to_label_studio_files(
    label_studio_folder: str, 
    conversations_file_path: str, 
    output_folder: str = None
):
    """
    Add conversation_id to Label Studio JSON files by matching text content from a conversations file.
    
    Args:
        label_studio_folder: Path to folder containing Label Studio JSON files
        conversations_file_path: Path to the file containing conversations with IDs
        output_folder: Path to save the modified files (if None, will modify in place)
        
    Returns:
        None (files are modified in place or saved to output_folder)
    """
    # Create a dictionary for fast text lookup
    text_to_conversation_id = {}
    
    # Load conversations file
    print(f"Loading conversations from {conversations_file_path}")
    with open(conversations_file_path, 'r', encoding='utf-8') as f:
        conversation_data = json.load(f)["data"]
        for conversation_datum in conversation_data:
            conversation_id = conversation_datum.get('conversation_id')
            
            # Skip if no conversation_id is found
            if not conversation_id:
                continue
            
            # Add each turn's text to the lookup dictionary
            conv_text = ""
            for turn in conversation_datum.get('conversation', []):
                # In the conversations file, content is under 'content'
                conv_text += turn.get('content', '')

            if conv_text in text_to_conversation_id:
                print("Duplicate text in conversation...")
            if conv_text:
                # Use the text as key to find the conversation_id later
                text_to_conversation_id[conv_text] = conversation_id
    
    print(f"Loaded {len(text_to_conversation_id)} conversations.")
    
    # If output_folder is provided but doesn't exist, create it
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each Label Studio file
    file_count = 0
    for filename in os.listdir(label_studio_folder):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(label_studio_folder, filename)
        
        try:
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modified = False
            
            # Process each annotation task
            for task in data.get('annotations', []):
                # Get the conversations in this task
                conversations = task.get('data', {}).get('conversation', [])
                
                # Add conversation_id to each turn based on text matching
                conv_text = ""
                for turn in conversations:
                    conv_text += turn.get('text', '')
                
                    
                # Try to find a matching conversation_id
                conversation_id = text_to_conversation_id.get(conv_text)
                
                if conversation_id:
                    for turn in conversations:
                        turn['conversation_id'] = conversation_id
                    modified = True
            
            # Save the modified file
            if modified:
                output_path = os.path.join(output_folder, filename) if output_folder else file_path
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                file_count += 1
                
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    print(f"Added conversation IDs to {file_count} files")


def run_automatic_analysis_v0(dirpath):

    dataset = run_test_cedric(dirpath)
    # # Load dataset
    # dataset = Dataset.load('data/sample120.json')

    automatic_variants = [
        "gpt4o-json",
        "gpt4o-free",
        "gpto3mini-json",
        "gpto3mini-free",
    ]

    # Load automatic annotations
    for variant in automatic_variants:
        print("\n" + variant + "\n")
        for fpath in listdir_nohidden(os.path.join(dirpath, f"automatic_annotations_v1/{variant}")):
            if "prompt_topic" in fpath:
                continue
            annotation_set = AnnotationSet.load_automatic(path=fpath, source=variant.replace("-", "_"), dataset_id_override="sample120")
            dataset.add_annotations(annotation_set)

    return dataset


def run_test_cedric(dirpath):
    dataset = Dataset.load(os.path.join(dirpath, "sample120.json"), "sample120") 

    # TODO: This adds conversation IDs to LabelStudio data and saves it to `data/labelstudio_outputs_wcids/`
    # Remove this when we have them in the LS outputs by default.
    add_conversation_ids_to_label_studio_files_v2(
        os.path.join(dirpath, "labelstudio_outputs_v2/"),
        os.path.join(dirpath, 'sample120.json'),
        os.path.join(dirpath, "labelstudio_outputs_wcids_v2/"),
    )

    # split annotations into two folders without any duplicate conversation IDs in each file.
    split_labelstudio_files_by_conversation_id_v2(
        input_folder=os.path.join(dirpath, "labelstudio_outputs_wcids_v2/"),
        output_folder1=os.path.join(dirpath, "labelstudio_outputs_split1_v2/"),
        output_folder2=os.path.join(dirpath, "labelstudio_outputs_split2_v2/"),
    )

    annotation_sets1 = load_labelstudio_v2(
        os.path.join(dirpath, "labelstudio_outputs_split1_v2"), 
        source="split1",
        dataset_id="sample120",
        level="message",
    )
    annotation_sets2 = load_labelstudio_v2(
        os.path.join(dirpath, "labelstudio_outputs_split2_v2"), 
        source="split2",
        dataset_id="sample120",
        level="message",
    )
    
    # Testing:
    # for task_key in annotation_sets1:
    #     num_annotations = len(annotation_sets1[task_key].annotations) + len(annotation_sets2[task_key].annotations)
    #     print(f"Total annotations for {task_key}: {num_annotations}\n")

    print("starttinginsdgljsdfjksdfjl")
    for task, annotation_set in annotation_sets1.items():
        dataset.add_annotations(annotation_set)
    for task, annotation_set in annotation_sets2.items():
        dataset.add_annotations(annotation_set)

    # Testing:
    # for task_key in annotation_sets1:
    #     a1_count = sum([1 if f"split1-{task_key}" in m.metadata else 0 for cc in dataset.data for m in cc.conversation])
    #     a2_count = sum([1 if f"split2-{task_key}" in m.metadata else 0 for cc in dataset.data for m in cc.conversation])
    #     print(f"Total annotations for {task_key}: {a1_count + a2_count}\n")

    return dataset

    # TODO: group the annotations

    # TODO: Cedric write this loader function from a folder / file of whatever type.
    # annotations = AnnotationSet.load_labelstudio("data/labelstudio_outputs_split1",)
    # annotations = AnnotationSet.load_labelstudio("data/labelstudio_outputs_split2",)

    # TODO: Zoey write this loader function from a folder / file of whatever type.
    # annotations = AnnotationSet.load_automatic("data/gpt_annotation_outputs.json")

    # dataset.add_annotations(annotations)

    # Get distribution for single feature
    # print("\nGetting annotation distributions...")
    # info_to_plot1a = dataset.get_annotation_distribution(name='<annotation_name>', level="conversation", annotation_source='<source_name>')
    
    # fig = barplot_distribution(
    #     info_to_plot1a, normalize=True, xlabel="X", ylabel="Proportion", title="Annotation Feature", 
    #     output_path="data/fig1.png")


def run_test():
    # Load the dataset
    dataset = Dataset.load('fake_data/fake_dataset.json')
    print(f"Loaded dataset with {len(dataset.data)} conversations")
    
    # Load annotation sets
    ann_human = AnnotationSet.load('fake_data/human_labels.json')
    ann_model_v1 = AnnotationSet.load('fake_data/model_v1_labels.json')
    
    # Add annotations to dataset
    dataset.add_annotations(ann_human)
    dataset.add_annotations(ann_model_v1)
    
    # Get distribution for single feature
    print("\nGetting annotation distributions...")
    info_to_plot1a = dataset.get_annotation_distribution(name='language', level="conversation", annotation_source='human_labels')
    info_to_plot1b = dataset.get_annotation_distribution(name='language', level="conversation", annotation_source='model_v1_labels')
    
    print("Human language annotations distribution:")
    print(info_to_plot1a)
    
    print("\nModel language annotations distribution:")
    print(info_to_plot1b)
    
    # Get model distribution
    model_distribution = dataset.get_annotation_distribution(name='model', level="conversation")
    print("\nModel distribution:")
    print(model_distribution)
    
    # Get joint distribution of model and language
    print("\nGetting joint distribution of model and language...")
    info_to_plot2a = dataset.get_joint_distribution(
        annotations1=('model', None), 
        annotations2=('language', 'model_v1_labels'), 
        level="conversation")
    
    print("Joint distribution matrix:")
    print(info_to_plot2a)
    
    # Compare human and model language annotations
    print("\nComparing human and model language annotations...")
    info_to_plot2b, agreement_metrics, disagreement_rows = dataset.get_joint_distribution(
        annotations1=('language', 'human_labels'), 
        annotations2=('language', 'model_v1_labels'), 
        level="conversation",
        compute_disagreement=True
    )
    
    print("Confusion matrix between human and model annotations:")
    print(info_to_plot2b)
    
    print("\nDisagreement rows:")
    for i, row in enumerate(disagreement_rows):
        print(f"Conversation {row[0]}: Human said '{row[1]}', Model said '{row[2]}'")
        if i > 10:
            break
    
    # Print agreement metrics
    tabulate_annotation_pair_summary(agreement_metrics)
    
    # Plot distributions
    print("\nPlotting distributions... (Not shown in console output)")
    # These would produce plots in a graphical environment
    fig = barplot_distribution(
        info_to_plot1a, normalize=True, xlabel="Languages", ylabel="Proportion", title="Languages (Human Annotations)", 
        output_path="fake_data/fig1.png")
    fig = barplot_distribution(
        info_to_plot1b, normalize=False, xlabel="Languages", ylabel="Frequency", title="Languages (Model v1 Annotations)",
        output_path="fake_data/fig2.png")
    fig = barplot_distribution(
        {"Human Annotations": info_to_plot1a, "Model v1 Annotations": info_to_plot1b}, normalize=True, 
        xlabel="Languages", ylabel="Proportion", title="Language Annotations",
        output_path="fake_data/fig3.png")
    fig = plot_confusion_matrix(
        info_to_plot2a, normalize=True, xlabel="Models", ylabel="Languages", title="Models v Languages",
        output_path="fake_data/fig4.png")
    fig = plot_confusion_matrix(
        info_to_plot2b, normalize=True, xlabel="Languages (Human Annotations)", 
        ylabel="Languages (Model v1 Annotations)", title="Language Annotations - Confusion Matrix",
        output_path="fake_data/fig5.png")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    run_test_cedric()