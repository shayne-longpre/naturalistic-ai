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
from lingua import LanguageDetectorBuilder, Language, LanguageDetector

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

def check_for_math(text, threshold = 3): 
    "This function checks if a give string contains >= threshold occurances of common mathematical patterns, such as math operations and symbols."
    math_patterns = [
        r'((?:[+\-]?\d*x\^\d+\s*)+)',         # matches sequences like x^5 + 3x^3 + x^2 + 2x
        r'\b[+\-]?\d*x\^\d+\b',               # matches terms like 2x^2, x^3, etc.
        r'\[\s*[a-z](?:\^\d+)?\s*\]',         # matches [x], [x^2], [2x], etc.
        r'\b[+\-]?\d+x\b',                    # terms like +2x, -4x, 1x, 3x, 9x, etc.
        r'[a-z]\([x-z]\)\s*=',                # function definitions like f(x)=, g(x)=, a(x)=
        r'\b[a-z]\s*[+\-*/=]\s*[a-z0-9]',     # variables with operators
        r'[a-z]\^[0-9]+',                     # exponents
        r'[+\-*/=]\s*[0-9a-z]',               # mathematical operators
    ]
    count = 0 
    for pattern in math_patterns:
        if re.search(pattern, text):
            count += 1

    if count >= threshold:
        return True
    
    return False
 

def predict_languages_lingua(text: str =  "Find all zeros in the indicated finite field of the given polynomial", detector: LanguageDetector = None): 
    """This function uses the Lingua library to detect languages in a given piece of text. 
       The input is a string to detect languages for, and the Lingua detector (passed in function to avoid loading the model for each example). 
       The function returns a de-deuplciated list of languages detected with a confidence above 0.05.
       
       ---Handling Math Equations---
       During initial testing, math equations were frequently misclassified as LATIN (e.g., "x^2 + 3x + 2 = 0"). To mitigate this, 
       when LATIN is detected by Lingua, this function splits the text into sentences, removes any sentence with a significant (>= 3) number of mathmatical patterns, 
       and rejoins the sentences. Once the text is rejoined, it checks the confidence of LATIN again. If the confidence is still above 0.05, it is considered a valid detection.
       If the confidence is reduced to below 0.05, the example is not labled as LATIN. 

       One consequence of this approach is that this function will be unable to detect Latin in a sentence with significant math. Our experience has been that most Latin examples 
       have at least one sentence without math, which would produce a correct classification. 
       """
       
    found_languages = []

    for result in detector.detect_multiple_languages_of(text):
        
        confidence = detector.compute_language_confidence(text, language=result.language)
        #print(f"Detected languages: {result.language.name} with confidence {confidence}, specifically the portion: {text[result.start_index:result.end_index]}")

        if confidence > 0.05:
            found_language = result.language.name
            if found_language == "LATIN":  
                segments = re.split(r'[.!?\n\t]+', text[result.start_index:result.end_index]) # split into sentences by punctuation, new line, or tabs. 
                non_math_segments = [segment for segment in segments if not check_for_math(segment)]
                non_math_combined = "\n".join(non_math_segments)
                latin_confidence = detector.compute_language_confidence(non_math_combined, language=result.language)
                if latin_confidence > 0.05:
                    found_languages.append(found_language)
                    if found_language != "ENGLISH": 
                        print(f"Detected language: {found_language} with confidence {confidence} for text: {text[result.start_index:result.end_index]}")
                        print("non math version: ", non_math_combined)
                    #print(f"Detected LATIN language with confidence {latin_confidence} after removing math segments.")
                else: 
                    #print(f"Detected LATIN language, but confidence {latin_confidence} is too low after removing math segments. Skipping: {text[result.start_index:result.end_index]}")
                    continue     
            else: 
                if found_language != "ENGLISH": 
                    print(f"Detected language: {found_language} with confidence {confidence} for text: {text[result.start_index:result.end_index]}")
                found_languages.append(found_language)
       
    found_languages = list(set(found_languages)) # de-duplicate the list of languages
    #print(f"RETURN: {found_languages}")
    return found_languages 


def run_simple_annotations(args, verbose = True):
    # Define metrics to compute with their functions
    metrics = {
        "text_length": get_string_length,
        "word_count": get_word_count,
        "char_count": get_char_count,
        "token_count": get_token_count,
        "languages": predict_languages_lingua
    }

    # Load tokenizer and language ID model
    tokenizer = tiktoken.encoding_for_model("gpt-4o") 
    language_detector = LanguageDetectorBuilder.from_languages(*Language.all()).build()

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
                    metric_value = predict_languages_lingua(text, language_detector)
                    annotator = "lingua"
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