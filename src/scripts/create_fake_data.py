import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Create a small dataset with fake conversations
def create_fake_dataset():
    # Create fake messages
    def create_fake_messages(num_messages=3):
        messages = []
        for i in range(num_messages):
            role = "human" if i % 2 == 0 else "assistant"
            content = f"This is a {role} message #{i//2 + 1}"
            timestamp = (datetime.now() - timedelta(minutes=num_messages-i)).isoformat()
            
            message = {
                "turn": i,
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "metadata": {}
            }
            messages.append(message)
        return messages
    
    # Create fake conversations
    conversations = []
    languages = ["English", "Spanish", "French", "German", "Chinese"]
    models = ["claude-3", "claude-3.5", "gpt-4"]
    geographies = ["US", "EU", "APAC", "LATAM"]
    
    for i in range(10):  # Create 10 conversations
        conversation_id = f"conv_{i:03d}"
        dataset_id = "test_dataset_001"
        user_id = f"user_{random.randint(1000, 9999)}"
        time = (datetime.now() - timedelta(days=i)).isoformat()
        model = random.choice(models)
        geography = random.choice(geographies)
        
        conversation = {
            "conversation_id": conversation_id,
            "dataset_id": dataset_id,
            "user_id": user_id,
            "time": time,
            "model": model,
            "conversation": create_fake_messages(random.randint(2, 6) * 2 - 1),  # Odd number of messages
            "geography": geography,
            "metadata": {}
        }
        conversations.append(conversation)
    
    # Create the dataset
    dataset = {
        "dataset_id": "test_dataset_001",
        "data": conversations
    }
    
    # Save the dataset to a file
    with open('fake_data/fake_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Created fake dataset with {len(conversations)} conversations")
    return dataset

# Create fake human annotations for language
def create_human_language_annotations(dataset):
    languages = ["English", "Spanish", "French", "German", "Chinese"]
    annotations = []
    
    for conversation in dataset["data"]:
        conv_id = conversation["conversation_id"]
        # Randomly assign a language
        language = random.choice(languages)
        annotations.append({
            "value": language,
            "target_id": conv_id
        })
    
    annotation_set = {
        "source": "human_labels",
        "name": "language",
        "level": "conversation",
        "dataset_id": dataset["dataset_id"],
        "annotations": annotations
    }
    
    # Save to file
    with open('fake_data/human_labels.json', 'w') as f:
        json.dump(annotation_set, f, indent=4)
    
    print(f"Created {len(annotations)} human language annotations")
    return annotation_set

# Create fake model annotations for language (slightly different from human annotations)
def create_model_language_annotations(dataset):
    languages = ["English", "Spanish", "French", "German", "Chinese"]
    annotations = []
    
    for conversation in dataset["data"]:
        conv_id = conversation["conversation_id"]
        
        # Get human annotation for this conversation to introduce some disagreement
        human_annotation = None
        with open('fake_data/human_labels.json', 'r') as f:
            human_data = json.load(f)
            for annotation in human_data["annotations"]:
                if annotation["target_id"] == conv_id:
                    human_annotation = annotation["value"]
                    break
        
        # 70% chance to agree with human, 30% chance to pick a different language
        if random.random() < 0.7 and human_annotation:
            language = human_annotation
        else:
            available_languages = [lang for lang in languages if lang != human_annotation]
            language = random.choice(available_languages)
            
        annotations.append({
            "value": language,
            "target_id": conv_id
        })
    
    annotation_set = {
        "source": "model_v1_labels",
        "name": "language",
        "level": "conversation",
        "dataset_id": dataset["dataset_id"],
        "annotations": annotations
    }
    
    # Save to file
    with open('fake_data/model_v1_labels.json', 'w') as f:
        json.dump(annotation_set, f, indent=4)
    
    print(f"Created {len(annotations)} model language annotations")
    return annotation_set

# Main function to create all the test data
def create_all_test_data():
    # Create and save the dataset
    dataset = create_fake_dataset()
    
    # Create and save the human annotations
    human_annotations = create_human_language_annotations(dataset)
    
    # Create and save the model annotations (with some disagreement)
    model_annotations = create_model_language_annotations(dataset)
    
    print("All test data created successfully!")
    print("Files generated:")
    print("- fake_dataset.json")
    print("- human_labels.json")
    print("- model_v1_labels.json")

# Run the main function
if __name__ == "__main__":
    create_all_test_data()