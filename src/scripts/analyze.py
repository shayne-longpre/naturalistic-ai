import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Callable
from datetime import datetime
import json
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, f1_score


class AnnotatedDataset:
    """
    A class to represent and analyze annotated conversations between users and AI chatbots.
    
    This class provides methods to:
    1. Load and filter conversation data
    2. Query annotations across different dimensions 
    3. Compare annotations from different sources
    4. Visualize distributions and agreement metrics
    """
    
    def __init__(self, data: Union[str, pd.DataFrame, List[Dict]]):
        """
        Initialize dataset from various input formats.
        
        Args:
            data: Path to a JSON file, a pandas DataFrame, or a list of dictionaries
        """
        if isinstance(data, str):
            with open(data, 'r') as f:
                self.data = pd.DataFrame(json.load(f))
        elif isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, list):
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("Data must be a path to a JSON file, a pandas DataFrame, or a list of dictionaries")
        
        # Ensure timestamps are datetime objects
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
    
    def filter(self, **kwargs) -> 'AnnotatedDataset':
        """
        Filter the dataset based on various criteria.
        
        Args:
            **kwargs: Filtering criteria (e.g., start_time, end_time, model_name, geography)
        
        Returns:
            A new AnnotatedDataset with filtered data
        """
        filtered_data = self.data.copy()
        
        # Time-based filtering
        if 'start_time' in kwargs and 'timestamp' in filtered_data.columns:
            start_time = pd.to_datetime(kwargs['start_time'])
            filtered_data = filtered_data[filtered_data['timestamp'] >= start_time]
            
        if 'end_time' in kwargs and 'timestamp' in filtered_data.columns:
            end_time = pd.to_datetime(kwargs['end_time'])
            filtered_data = filtered_data[filtered_data['timestamp'] <= end_time]
        
        # Other column-based filtering
        for key, value in kwargs.items():
            if key not in ['start_time', 'end_time'] and key in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[key] == value]
        
        return AnnotatedDataset(filtered_data)
    
    def get_annotation_distribution(self, annotation_name: str, 
                                   group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate the distribution of a specific annotation.
        
        Args:
            annotation_name: The name of the annotation column to analyze
            group_by: Optional list of columns to group by (e.g., ['model_name', 'geography'])
        
        Returns:
            DataFrame with distribution statistics
        """
        if annotation_name not in self.data.columns:
            raise ValueError(f"Annotation '{annotation_name}' not found in the dataset")
        
        if group_by is None:
            # Simple distribution across the entire dataset
            return pd.DataFrame(self.data[annotation_name].value_counts(normalize=True)).reset_index().rename(
                columns={'index': annotation_name, annotation_name: 'frequency'})
        else:
            # Grouped distribution
            result = []
            for group_name, group_data in self.data.groupby(group_by):
                dist = group_data[annotation_name].value_counts(normalize=True).reset_index().rename(
                    columns={'index': annotation_name, annotation_name: 'frequency'})
                
                # Add group information
                if isinstance(group_name, tuple):
                    for i, group_col in enumerate(group_by):
                        dist[group_col] = group_name[i]
                else:
                    dist[group_by[0]] = group_name
                    
                result.append(dist)
            
            return pd.concat(result, ignore_index=True)
    
    def compare_annotations(self, annotation1: str, annotation2: str, 
                           filter_criteria: Optional[Dict] = None) -> Dict:
        """
        Compare two annotation sets and calculate agreement metrics.
        
        Args:
            annotation1: First annotation column name
            annotation2: Second annotation column name
            filter_criteria: Optional dictionary to filter data before comparison
        
        Returns:
            Dictionary with comparison metrics and confusion data
        """
        if annotation1 not in self.data.columns or annotation2 not in self.data.columns:
            raise ValueError(f"One or both annotations not found in the dataset")
        
        # Apply filters if provided
        dataset = self if filter_criteria is None else self.filter(**filter_criteria)
        
        # Get data points where both annotations exist
        valid_data = dataset.data.dropna(subset=[annotation1, annotation2])
        
        if len(valid_data) == 0:
            return {"error": "No valid data points with both annotations"}
        
        # Extract the labels
        labels1 = valid_data[annotation1]
        labels2 = valid_data[annotation2]
        
        # Get unique labels from both annotations
        all_labels = sorted(list(set(labels1.unique()) | set(labels2.unique())))
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels1, labels2, labels=all_labels)
        
        # Calculate agreement metrics
        try:
            kappa = cohen_kappa_score(labels1, labels2)
        except Exception:
            kappa = np.nan
            
        accuracy = accuracy_score(labels1, labels2)
        
        # Identify disagreements
        disagreements = valid_data[labels1 != labels2].copy()
        
        # Construct result
        result = {
            "confusion_matrix": cm,
            "labels": all_labels,
            "metrics": {
                "cohen_kappa": kappa,
                "accuracy": accuracy,
                "total_samples": len(valid_data),
                "agreement_count": sum(labels1 == labels2),
                "disagreement_count": sum(labels1 != labels2),
                "disagreement_rate": sum(labels1 != labels2) / len(valid_data)
            },
            "disagreements": disagreements
        }
        
        return result

    def plot_annotation_distribution(self, annotation_name: str, 
                                    group_by: Optional[Union[str, List[str]]] = None,
                                    title: Optional[str] = None,
                                    figsize: Tuple[int, int] = (10, 6),
                                    rotate_labels: bool = False) -> None:
        """
        Plot the distribution of a specific annotation.
        
        Args:
            annotation_name: The name of the annotation to plot
            group_by: Optional column(s) to group by
            title: Optional plot title
            figsize: Figure size tuple
            rotate_labels: Whether to rotate x-axis labels
        """
        if group_by is not None and isinstance(group_by, str):
            group_by = [group_by]
            
        distribution = self.get_annotation_distribution(annotation_name, group_by)
        
        plt.figure(figsize=figsize)
        
        if group_by is None:
            # Simple bar chart
            sns.barplot(x=annotation_name, y='frequency', data=distribution)
            if title is None:
                title = f"Distribution of {annotation_name}"
        else:
            # Grouped bar chart
            if len(group_by) == 1:
                # One grouping variable
                sns.barplot(x=annotation_name, y='frequency', hue=group_by[0], data=distribution)
                if title is None:
                    title = f"Distribution of {annotation_name} by {group_by[0]}"
            else:
                # Multiple grouping variables - create a combined group column
                distribution['group'] = distribution.apply(
                    lambda row: ' - '.join(str(row[col]) for col in group_by), axis=1)
                sns.barplot(x=annotation_name, y='frequency', hue='group', data=distribution)
                if title is None:
                    title = f"Distribution of {annotation_name} by {', '.join(group_by)}"
        
        plt.title(title)
        plt.xlabel(annotation_name)
        plt.ylabel('Frequency')
        
        if rotate_labels:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        plt.show()
        
    def plot_annotation_comparison(self, annotation1: str, annotation2: str,
                                 filter_criteria: Optional[Dict] = None,
                                 title: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 8),
                                 normalize: bool = True,
                                 annotate: bool = True) -> None:
        """
        Plot a heatmap comparison between two annotation sets.
        
        Args:
            annotation1: First annotation column name (displayed on rows)
            annotation2: Second annotation column name (displayed on columns)
            filter_criteria: Optional dictionary to filter data before comparison
            title: Optional plot title
            figsize: Figure size tuple
            normalize: Whether to normalize the confusion matrix by rows
            annotate: Whether to annotate cells with values
        """
        comparison_data = self.compare_annotations(annotation1, annotation2, filter_criteria)
        
        if "error" in comparison_data:
            print(f"Error: {comparison_data['error']}")
            return
            
        cm = comparison_data["confusion_matrix"]
        labels = comparison_data["labels"]
        metrics = comparison_data["metrics"]
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Replace NaN with 0
            
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=annotate, fmt='.2f' if normalize else 'd',
                  xticklabels=labels, yticklabels=labels, cmap="Blues")
        
        if title is None:
            title = f"Comparison of {annotation1} vs {annotation2}"
            
        subtitle = (f"Accuracy: {metrics['accuracy']:.2f}, "
                   f"Cohen's Kappa: {metrics['cohen_kappa']:.2f}, "
                   f"Disagreement Rate: {metrics['disagreement_rate']:.2f}")
        
        plt.title(f"{title}\n{subtitle}")
        plt.xlabel(annotation2)
        plt.ylabel(annotation1)
        plt.tight_layout()
        plt.show()
        
    def save_disagreements(self, annotation1: str, annotation2: str, 
                         output_path: str,
                         filter_criteria: Optional[Dict] = None) -> None:
        """
        Save disagreements between two annotation sets to a file.
        
        Args:
            annotation1: First annotation column name
            annotation2: Second annotation column name
            output_path: Path to save the disagreements CSV
            filter_criteria: Optional dictionary to filter data before comparison
        """
        comparison_data = self.compare_annotations(annotation1, annotation2, filter_criteria)
        
        if "error" in comparison_data:
            print(f"Error: {comparison_data['error']}")
            return
            
        disagreements = comparison_data["disagreements"]
        disagreements.to_csv(output_path, index=False)
        print(f"Saved {len(disagreements)} disagreements to {output_path}")
        
    def custom_query(self, query_function: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """
        Apply a custom query function to the dataset.
        
        Args:
            query_function: A function that takes a DataFrame and returns a modified DataFrame
            
        Returns:
            The result of the custom query
        """
        return query_function(self.data)


# Utility functions for batch processing and multi-dataset comparisons

def compare_multiple_annotations(dataset: AnnotatedDataset, 
                               annotation_sets: List[Tuple[str, str]],
                               filter_criteria: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    Compare multiple pairs of annotations.
    
    Args:
        dataset: The AnnotatedDataset to analyze
        annotation_sets: List of (annotation1, annotation2) pairs to compare
        filter_criteria: Optional dictionary to filter data before comparison
        
    Returns:
        Dictionary mapping annotation pairs to comparison results
    """
    results = {}
    
    for annotation1, annotation2 in annotation_sets:
        pair_name = f"{annotation1}_vs_{annotation2}"
        results[pair_name] = dataset.compare_annotations(annotation1, annotation2, filter_criteria)
        
    return results

def plot_multi_distribution(datasets: List[Tuple[AnnotatedDataset, str]], 
                          annotation_name: str,
                          title: str = None,
                          figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot annotation distribution across multiple datasets.
    
    Args:
        datasets: List of (dataset, name) tuples
        annotation_name: The annotation to analyze
        title: Optional plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    all_distributions = []
    for dataset, name in datasets:
        dist = dataset.get_annotation_distribution(annotation_name)
        dist['source'] = name
        all_distributions.append(dist)
        
    combined_dist = pd.concat(all_distributions, ignore_index=True)
    
    sns.barplot(x=annotation_name, y='frequency', hue='source', data=combined_dist)
    
    if title is None:
        title = f"Distribution of {annotation_name} Across Datasets"
        
    plt.title(title)
    plt.xlabel(annotation_name)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title="Dataset")
    plt.show()