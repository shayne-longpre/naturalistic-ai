import fasttext 
import sys
import argparse
import pandas as pd
import tiktoken
sys.path.append("./")
sys.path.append("./src") 
import re
import fasttext
from huggingface_hub import hf_hub_download
import os 
from src.classes.annotation_record import AnnotationRecord
from src.classes.annotation_set import AnnotationSet

sys.path.append("./")

def get_string_length(x): 
    return len(x)

def get_word_count(x): 
    return len(x.split())

def get_char_count(x):
    return len(x.replace(" ", ""))

def get_token_count(x, tokenizer): 
    enc = tokenizer.encode(x)
    return len(enc)


def predict_languages(text, model): 
    """ Predict languages in a given text using a fasttext language identification model.
    This function splits the text into sentences and predicts the language for each sentence. 
    All predicted languages are returned as a non-duplicated list. 
    If no language is detected, ["unknown"] is returned.
    """
    
    languages = []
    # Split text by sentence-ending punctuation: . ? !
    sentences = re.split(r'[.!?]\s*', text)
    sentences = [s for s in sentences if s.strip()]  # Remove empty strings

    for sentence in sentences:
        try:
            #lang = detect(sentence)
            lang = [x[0].replace("__label__", "").replace("_Latn", "") for x in model.predict([sentence], k=1)[0]][0]
        except:
            lang = "unknown"
        languages.append(lang)
    
    # Return unique languages detected in the text
    languages = list(set(languages))
    if len(languages) == 0:
        languages.append("unknown")
    return languages 


def run_simple_annotations(args, verbose = True):
    # Define metrics to compute with their functions
    metrics = {
        "text_length": get_string_length,
        "word_count": get_word_count,
        "char_count": get_char_count,
        "token_count": get_token_count,
        "language": predict_languages
    }

    # Load tokenizer and language ID model
    tokenizer = tiktoken.encoding_for_model("gpt-4o") 
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin") 
    language_model = fasttext.load_model(model_path)  

    # Load dataset    
    dataframe = pd.read_json(args.input, orient="records")
    annotatation_sets = []

    # For each metric, iterate each conversation in the dataset, iterate each turn, compute the metric, and save the file. 
    # An annotation file is created for each metric. 
    for metric_name, metric_function in metrics.items():
        if verbose:
            print(f"Processing metric: {metric_name}")
        
        annotations = []

        for _, row in dataframe.iterrows():
            conversation = row["data"]
            coversation_id = conversation.get("conversation_id", "")
            for turn in conversation.get("conversation", []):
                target_id = f"{coversation_id}-{turn.get('turn', '')}"
                
                text = turn.get("content", "")

                # If the metric is token count, pass in the tokenizer. 
                # This is much more efficient than loading the tokenizer for each sample.
                annotator = ""
                if "token" in metric_name:
                    metric_value = metric_function(text, tokenizer)
                    annotator = "tiktoken"
                elif "language" in metric_name:
                    # If the metric is predicting languages, pass in the language ID model. 
                    metric_value = predict_languages(text, language_model)
                    annotator = "fasttext"
                else:
                    metric_value = metric_function(text)

                # Create AnnotationRecord for each metric
                annotation = AnnotationRecord(
                    value=metric_value, 
                    confidence=1.0, 
                    target_id=target_id, 
                    annotator=annotator
                )
                    
                annotations.append(annotation)
        
        # Create AnnotationSet object, and save it to a file
        annotation_set = AnnotationSet(
            source = "simple",
            name =metric_name,
            level = "message",
            dataset_id = dataframe.iloc[0]["dataset_id"],
            annotations = annotations
        )

        save_path = args.save.replace(".jsonl", f"_{metric_name}.jsonl")
        print(f"Saving annotations to {save_path}")
        annotation_set.save_to_json(save_path)
        annotatation_sets.append(annotation_set)

    return annotatation_sets # Return the list of AnnotationSet objects if needed 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input json file (original dataset).")
    parser.add_argument("--input_format", type=str, required=True, default=None)
    parser.add_argument("--save", type=str, required=True, help="Save path (will be adapted with the feature name).")
    args = parser.parse_args()

    run_simple_annotations(args)