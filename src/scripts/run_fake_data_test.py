import sys
import pandas as pd 
import json

sys.path.append("./")

from src.helpers.io import read_jsonl, write_json
from src.classes.dataset import Dataset
from src.classes.annotation_set import AnnotationSet
from src.helpers.visualisation import tabulate_interrater_metrics, barplot_distribution, plot_confusion_matrix


def run_test_cedric_zoey():
    dataset = Dataset.load('data/sample120.json')

    # TODO: Cedric write this loader function from a folder / file of whatever type.
    annotations = AnnotationSet.load_labelstudio("data/labelstudio_outputs.json")
    # TODO: Zoey write this loader function from a folder / file of whatever type.
    annotations = AnnotationSet.load_automatic("data/gpt_annotation_outputs.json")

    dataset.add_annotations(annotations)

    # Get distribution for single feature
    print("\nGetting annotation distributions...")
    info_to_plot1a = dataset.get_annotation_distribution(name='<annotation_name>', level="conversation", annotation_source='<source_name>')
    
    fig = barplot_distribution(
        info_to_plot1a, normalize=True, xlabel="X", ylabel="Proportion", title="Annotation Feature", 
        output_path="data/fig1.png")


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
        annotations1=('model', None), annotations2=('language', 'model_v1_labels'), level="conversation")
    
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
    tabulate_interrater_metrics(agreement_metrics)
    
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
    run_test_cedric_zoey()