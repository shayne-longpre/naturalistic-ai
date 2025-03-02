import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from analyze import AnnotatedDataset, compare_multiple_annotations, plot_multi_distribution

# Example 1: Basic Usage - Analyzing language distribution by time period
def example_language_distribution():
    # Sample data creation (in real usage, you'd load from files)
    np.random.seed(42)
    
    # Create sample conversation data
    num_samples = 1000
    languages = ['English', 'Spanish', 'French', 'German', 'Mandarin', 'Japanese']
    models = ['GPT-4', 'Claude', 'LLaMA']
    regions = ['North America', 'Europe', 'Asia', 'South America']
    
    sample_data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_samples):
        # Random timestamp within a 90-day period
        days_offset = np.random.randint(0, 90)
        sample_timestamp = start_date + timedelta(days=days_offset)
        
        # Create sample conversation with annotations
        conversation = {
            'conversation_id': f'conv_{i}',
            'user_id': f'user_{np.random.randint(1, 100)}',
            'model_name': np.random.choice(models),
            'timestamp': sample_timestamp,
            'geography': np.random.choice(regions),
            'language': np.random.choice(languages, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),  # Weighted distribution
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], p=[0.6, 0.3, 0.1]),
            'topic': np.random.choice(['tech', 'finance', 'health', 'entertainment', 'other']),
            'turns': np.random.randint(2, 15)
        }
        sample_data.append(conversation)
    
    # Create dataset
    dataset = AnnotatedDataset(sample_data)
    
    # Example 1: What is the distribution of languages in conversations between certain dates?
    print("Example 1: Language distribution analysis")
    print("-----------------------------------------")
    
    # Filter by time period
    q1_data = dataset.filter(
        start_time=datetime(2023, 1, 1),
        end_time=datetime(2023, 1, 31)
    )
    
    q2_data = dataset.filter(
        start_time=datetime(2023, 2, 1),
        end_time=datetime(2023, 3, 1)
    )
    
    # Plot language distribution for different time periods
    plt.figure(figsize=(14, 6))
    
    # Create subplots
    plt.subplot(1, 2, 1)
    q1_dist = q1_data.get_annotation_distribution('language')
    plt.bar(q1_dist['language'], q1_dist['frequency'])
    plt.title('Language Distribution (Jan 2023)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    q2_dist = q2_data.get_annotation_distribution('language')
    plt.bar(q2_dist['language'], q2_dist['frequency'])
    plt.title('Language Distribution (Feb 2023)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('language_distribution_by_period.png')
    plt.close()
    
    # Alternatively, use the built-in plotting function with grouping
    # Filter by time and group by model
    jan_feb_data = dataset.filter(
        start_time=datetime(2023, 1, 1),
        end_time=datetime(2023, 2, 28)
    )
    
    jan_feb_data.plot_annotation_distribution('language', group_by='model_name', 
                                             title="Language Distribution by Model (Jan-Feb 2023)",
                                             rotate_labels=True)
    
    # Compare datasets directly
    plot_multi_distribution(
        [(q1_data, "January"), (q2_data, "February")],
        'language',
        "Language Distribution Comparison by Month"
    )
    
    return dataset

# Example 2: Comparing annotations from different sources
def example_annotation_comparison(dataset):
    print("\nExample 2: Annotation comparison")
    print("-------------------------------")
    
    # Create a second annotation set (simulating automatic annotation)
    np.random.seed(100)
    
    # Add a second language annotation with some disagreement
    dataset.data['language_auto'] = dataset.data['language'].copy()
    
    # Introduce some errors (20% disagreement rate)
    error_indices = np.random.choice(len(dataset.data), size=int(len(dataset.data) * 0.2), replace=False)
    languages = ['English', 'Spanish', 'French', 'German', 'Mandarin', 'Japanese']
    
    for idx in error_indices:
        current_language = dataset.data.at[idx, 'language']
        other_languages = [lang for lang in languages if lang != current_language]
        dataset.data.at[idx, 'language_auto'] = np.random.choice(other_languages)
    
    # Compare the gold (human) labels with automatic labels
    comparison = dataset.compare_annotations('language', 'language_auto')
    
    print(f"Agreement metrics:")
    print(f"  - Accuracy: {comparison['metrics']['accuracy']:.4f}")
    print(f"  - Cohen's Kappa: {comparison['metrics']['cohen_kappa']:.4f}")
    print(f"  - Total samples: {comparison['metrics']['total_samples']}")
    print(f"  - Agreement count: {comparison['metrics']['agreement_count']}")
    print(f"  - Disagreement count: {comparison['metrics']['disagreement_count']}")
    print(f"  - Disagreement rate: {comparison['metrics']['disagreement_rate']:.4f}")
    
    # Plot the comparison heatmap
    dataset.plot_annotation_comparison('language', 'language_auto', 
                                     title="Human vs. Automatic Language Detection")
    
    # Save disagreements to CSV
    dataset.save_disagreements('language', 'language_auto', 'language_disagreements.csv')
    
    # Compare multiple annotation sets
    dataset.data['sentiment_auto'] = dataset.data['sentiment'].copy()
    # Introduce some errors in sentiment detection
    error_indices = np.random.choice(len(dataset.data), size=int(len(dataset.data) * 0.3), replace=False)
    sentiments = ['positive', 'neutral', 'negative']
    
    for idx in error_indices:
        current_sentiment = dataset.data.at[idx, 'sentiment']
        other_sentiments = [sent for sent in sentiments if sent != current_sentiment]
        dataset.data.at[idx, 'sentiment_auto'] = np.random.choice(other_sentiments)
    
    # Compare multiple annotation pairs
    multi_comparison = compare_multiple_annotations(
        dataset,
        [('language', 'language_auto'), ('sentiment', 'sentiment_auto')]
    )
    
    print("\nMultiple annotation comparisons:")
    for pair_name, results in multi_comparison.items():
        print(f"  - {pair_name}: Accuracy = {results['metrics']['accuracy']:.4f}, "
              f"Kappa = {results['metrics']['cohen_kappa']:.4f}")
    
    return dataset

# Example 3: Advanced custom queries and analysis
def example_custom_analysis(dataset):
    print("\nExample 3: Custom analysis")
    print("-------------------------")
    
    # Define a custom query function
    def complex_analysis(df):
        """
        Custom query: Find the relationship between conversation length (turns)
        and sentiment across different models and regions.
        """
        # Group by model, region, and sentiment
        grouped = df.groupby(['model_name', 'geography', 'sentiment'])
        
        # Calculate average turns and count
        analysis = grouped.agg({
            'turns': ['mean', 'count'],
            'user_id': 'nunique'  # Unique users per group
        }).reset_index()
        
        # Flatten the column names
        analysis.columns = ['model_name', 'geography', 'sentiment', 
                           'avg_turns', 'conversation_count', 'unique_users']
        
        return analysis.sort_values(['model_name', 'geography', 'sentiment'])
    
    # Run the custom analysis
    result = dataset.custom_query(complex_analysis)
    
    print("Sample of custom analysis results:")
    print(result.head())
    
    # Create a visualization from the custom analysis
    plt.figure(figsize=(14, 8))
    
    # Filter for clarity in the visualization
    plot_data = result[result['model_name'].isin(['GPT-4', 'Claude'])]
    
    # Create a grouped bar chart
    sns.barplot(x='geography', y='avg_turns', hue='sentiment', 
               col='model_name', data=plot_data, kind='bar',
               palette={'positive': 'green', 'neutral': 'gray', 'negative': 'red'})
    
    plt.title('Average Conversation Length by Geography, Sentiment, and Model')
    plt.xlabel('Geography')
    plt.ylabel('Average Turns')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('custom_analysis.png')
    plt.close()
    
    return result

if __name__ == "__main__":
    # Run the examples
    dataset = example_language_distribution()
    example_annotation_comparison(dataset)
    example_custom_analysis(dataset)
    
    print("\nAll examples complete. Check the output files for visualizations.")