import os
import sys
import typing
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from collections import Counter, defaultdict

sys.path.append("./")

from src.helpers.constants import FUNCTION_ANNOTATION_LABEL_ABBREVIATIONS


def barplot_distribution(
    data: typing.Union[typing.Dict[str, int], typing.Dict[str, typing.Dict[str, int]]],
    normalize: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    figsize: typing.Tuple[int, int] = (10, 6),
    output_path: typing.Optional[str] = None,
    order: typing.Optional[typing.Union[str, list]] = "descending",
    max_labels: int = 20
) -> plt.Figure:
    """
    Plot a bar chart of annotation distribution.
    
    Args:
        data: Either a single distribution dict or a dict of dicts for comparison
        normalize: Whether to normalize the values to proportions
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        figsize: Figure size (width, height) in inches
        output_path: If provided, save the figure to this path
        order: Order of x-axis categories - "ascending", "descending", or list of labels
        max_labels: Maximum number of x-axis labels to display

    Returns:
        The Matplotlib Figure object.
    """
    # Create figure with "constrained" layout
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    # Check if we have a single distribution or multiple
    if all(isinstance(v, (int, float)) for v in data.values()):
        # Single distribution
        categories = list(data.keys())
        values = list(data.values())
        
        if normalize:
            total = sum(values)
            values = [v / total for v in values]
        
        # Handle ordering
        if order == "ascending":
            sorted_data = sorted(zip(categories, values), key=lambda x: x[1])
            categories, values = zip(*sorted_data) if sorted_data else ([], [])
        elif order == "descending":
            sorted_data = sorted(zip(categories, values), key=lambda x: x[1], reverse=True)
            categories, values = zip(*sorted_data) if sorted_data else ([], [])
        elif isinstance(order, list):
            # Reorder according to provided list
            ordered_data = {k: data.get(k, 0) for k in order if k in data}
            categories = list(ordered_data.keys())
            values = list(ordered_data.values())
            if normalize and values:
                total = sum(values)
                values = [v / total for v in values]
            
        ax.bar(categories, values)
        
    else:
        # Multiple distributions for comparison
        # Convert to DataFrame for easier plotting
        dfs = []
        for source, distribution in data.items():
            df = pd.DataFrame(list(distribution.items()), columns=['Category', 'Count'])
            df['Source'] = source
            
            if normalize:
                df['Count'] = df['Count'] / df['Count'].sum()
                
            dfs.append(df)
            
        all_data = pd.concat(dfs)
        
        # Handle ordering for the DataFrame case
        if order == "ascending":
            # Get total count per category across all sources
            category_totals = all_data.groupby('Category')['Count'].sum().reset_index()
            order_list = category_totals.sort_values('Count')['Category'].tolist()
            sns.barplot(x='Category', y='Count', hue='Source', data=all_data, ax=ax, order=order_list)
        elif order == "descending":
            category_totals = all_data.groupby('Category')['Count'].sum().reset_index()
            order_list = category_totals.sort_values('Count', ascending=False)['Category'].tolist()
            sns.barplot(x='Category', y='Count', hue='Source', data=all_data, ax=ax, order=order_list)
        elif isinstance(order, list):
            valid_order = [cat for cat in order if cat in all_data['Category'].unique()]
            sns.barplot(x='Category', y='Count', hue='Source', data=all_data, ax=ax, order=valid_order)
        else:
            sns.barplot(x='Category', y='Count', hue='Source', data=all_data, ax=ax)
            
        ax.legend(title='')  # Remove legend title if desired
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=90)
    
    # Limit the number of x-axis labels if needed
    if len(ax.get_xticklabels()) > max_labels:
        # Keep only max_labels number of labels
        n_labels = len(ax.get_xticklabels())
        keep_indices = np.linspace(0, n_labels-1, max_labels, dtype=int)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticks([i for i in keep_indices])
        ax.set_xticklabels([labels[i] for i in keep_indices])
    
    # Save the plot if an output path is provided
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
    
    # Return the figure but prevent automatic display in Jupyter
    plt.close(fig)
    return fig


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    normalize: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    figsize: typing.Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    output_path: typing.Optional[str] = None
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        matrix: DataFrame containing the confusion matrix
        normalize: Whether to normalize by rows
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        figsize: Figure size (width, height) in inches
        cmap: Colormap for the heatmap
        output_path: If provided, save the figure to this path

    Returns:
        The Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    
    # Plot heatmap
    sns.heatmap(matrix, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, ax=ax)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # fig.tight_layout()
    
    # Save the figure if an output path is provided
    if output_path is not None:
        fig.savefig(output_path)

    # Return the figure but prevent automatic display in Jupyter
    plt.close(fig)
    return fig



def analyze_pair_annotations(
    annotations: typing.List[typing.Tuple[str, typing.Any, typing.Any]],
    order_matters: bool = True,
) -> pd.DataFrame:
    """
    Analyze pairs of annotations and produce a CSV report in descending order of frequency.
    
    Parameters:
    -----------
    annotations : List[Tuple[str, Any, Any]]
        List of tuples in the format (uid, val1, val2) where val1 and val2 can be 
        ints, strings, bools, or lists
    output_file : str, optional
        Path to the output CSV file, default is "annotation_analysis.csv"
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the analysis results
    """
    # Convert annotations for counting
    normalized_pairs = []
    for uid, val1, val2 in annotations:
        norm_val1 = tuple(sorted(val1)) if isinstance(val1, list) else val1
        norm_val2 = tuple(sorted(val2)) if isinstance(val2, list) else val2
        if order_matters:  # If we do care which side is which:
            normalized_pairs.append((uid, norm_val1, norm_val2))
        else:  # If we don't care which side is which:
            normalized_pairs.append((uid, norm_val1, norm_val2) if norm_val1 < norm_val2 else (uid, norm_val2, norm_val1))
        
    
    # Count the occurrences of each pair
    pair_counts = Counter([(v1, v2) for _, v1, v2 in normalized_pairs])
    
    # Create result data
    results = []
    for (val1, val2), count in pair_counts.most_common():
        results.append({
            'val1': list(val1) if isinstance(val1, tuple) else val1,
            'val2': list(val2) if isinstance(val2, tuple) else val2,
            'count': count,
            'percent_of_all': round((count / len(annotations)) * 100, 2),
            'matching': val1 == val2
        }) 
    return pd.DataFrame(results)


def tabulate_annotation_pair_summary(
    df: pd.DataFrame, 
    num_rows: int = None,
    max_chars: int = 30
) -> str:
    """
    Format a DataFrame for display with controlled number of rows and shortened values.
    
    Parameters:
    -----------
    df: DataFrame containing the annotation analysis results
    num_rows : int, optional—Number of rows to display. If None, display all rows
    max_chars : int, optional
        Maximum number of characters allowed for val1/val2 before shortening.
        If a value exceeds this length, each word will be shortened to its first letter.
    """
    display_df = df.copy()
    
    def shorten_value(val):
        # For lists, calculate the total length of all elements
        if isinstance(val, list) and sum(len(str(item)) for item in val) > max_chars:
            # Shorten each word in each item to its first letter
            ret_val = []
            for item in val:
                abbrev = FUNCTION_ANNOTATION_LABEL_ABBREVIATIONS.get(item.lower())
                abbrev_backup = ' '.join(word[:3] for word in str(item).split())
                ret_val.append(abbrev if abbrev is not None else abbrev_backup)
            return ret_val
        elif isinstance(val, str) and len(val) > max_chars:
            return ' '.join(word[:4] for word in val.split())
        return val
    
    # Apply shortening to val1 and val2 columns
    display_df['val1'] = display_df['val1'].apply(shorten_value)
    display_df['val2'] = display_df['val2'].apply(shorten_value)
    
    # Limit rows if specified
    if num_rows is not None:
        display_df = display_df.head(num_rows)
    
    # Format boolean values as "T"/"F" for better readability
    display_df['matching'] = display_df['matching'].map({True: 'T', False: 'F'})
    return tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)


def run_interrater_comparison(
    dset, 
    task_name,
    annotation_source_1,
    annotation_source_2,
    outdirpath,
):
    info_to_plot1 = dset.get_annotation_distribution(name=task_name, level="message", annotation_source=annotation_source_1)
    info_to_plot2 = dset.get_annotation_distribution(name=task_name, level="message", annotation_source=annotation_source_2)
    info_to_plot1b = dset.get_annotation_distribution(name=task_name, level="message", annotation_source=annotation_source_1, annotation_as_list_type=True)
    info_to_plot2b = dset.get_annotation_distribution(name=task_name, level="message", annotation_source=annotation_source_2, annotation_as_list_type=True)

    outdir = os.path.join(outdirpath, "{annotation_source_1}--{annotation_source_2}/{task_name}")
    os.makedirs(outdir, exist_ok=True)
    fig = barplot_distribution(
        {"Split1": info_to_plot1, "Split2": info_to_plot2}, normalize=True, 
        xlabel=task_name, ylabel="Proportion", title="",
        output_path=f"{outdir}/barchart.png", order="descending")
    
    fig_b = barplot_distribution(
        {"Split1": info_to_plot1b, "Split2": info_to_plot2b}, normalize=True, 
        xlabel=task_name, ylabel="Proportion", title="",
        output_path=f"{outdir}/multilabel_barchart.png", order="descending")

    info_to_plot_cm, agreement_metrics, paired_values = dset.get_joint_distribution(
        annotations1=(task_name, annotation_source_1), 
        annotations2=(task_name, annotation_source_2), 
        level="message",
        compute_disagreement=True,
        verbose=True
    )
    # print(info_to_plot_cm)

    fig2 = plot_confusion_matrix(info_to_plot_cm, normalize=True, xlabel="", ylabel="", title="Confusion Matrix", output_path=f"{outdir}/confusion_matrix.png")

    # print(paired_values[0:3])
    df = analyze_pair_annotations(paired_values, order_matters=True)
    df.to_csv(f"{outdir}/pair_frequencies.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    print()
    print(f"-----------------{task_name}-----------------")
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in agreement_metrics.items()})
    print(tabulate_annotation_pair_summary(df, 20))
    print(len(df))
    print()
    return paired_values


def display_info_for_turn(
    dset,
    ex_idx_turn,
    relevant_keys,
):
    """For a given dataset and example ID, print out the details and annotations for `relevant_keys`."""

    ex_idx, turn = ex_idx_turn.split("-")
    turn = int(turn)
    message = dset.id_lookup(ex_idx_turn, level="message")[ex_idx_turn].to_dict()
    role = message['role']
    # relevant_keys = prompt_fields_new if role == "user" else response_fields_new
    task_to_source_to_vals = defaultdict(dict)
    for key in message["metadata"].keys():
        source, task = key.split("-")
        if task in relevant_keys:
            task_to_source_to_vals[task][source] = message["metadata"][key]

    print(f"IDX: {ex_idx} | Turn: {turn} | Role: {role}")
    print(f"-------------------------------------------")
    for task, source_vals in task_to_source_to_vals.items():
        print()
        print(f"TASK: {task}")
        for source, val in source_vals.items():
            src_info = val["annotator"] if "split" in source else source
            print(f"{src_info}:   {val['value']}")

    print("\n****** Message Content:******")
    print(message["content"])
    print()

    if turn > 0:
        print("\n****** Previous Turn Message Content:******")
        prev_message = dset.id_lookup(ex_idx + "-" + str(turn-1), level="message")[ex_idx + "-" + str(turn-1)].to_dict()
        print(prev_message["content"])

    

def plot_differences_for_group(
    group_name: str,
    group_diff_data: dict,
    baseline_label: str,
    comparison_label: str,
    outdir: str = "data/annotation_analysis_v0/data-slice-comparison"
):
    """
    Plot and save diverging bar charts comparing a group's annotation differences to the baseline.

     Args:
        group_name (str): Name of the group.
        group_diff_data (dict): Result of compare_annotations_to_baseline()[group_name],
            where keys are (attribute_name, source) and values have:
                "differences": [(label, diff, group_pct, base_pct), …]
                "metrics": {statistic_name: value, …}
        baseline_label (str): Name(s) of baseline.
        comparison_label (str): Name(s) of comparison group.
        outdir (str): Directory to save plots.
    """
    os.makedirs(outdir, exist_ok=True)

    for (attribute_name, source), result in group_diff_data.items():
        diffs = result["differences"]
        metrics = result["metrics"]

        if not diffs:
            print(f"[Skipped] No data available to plot for {attribute_name} ({source})")
            continue

        labels, differences, group_pcts, base_pcts = zip(*diffs)

        colors = ["green" if diff > 0 else "red" for diff in differences]

        plt.figure(figsize=(12, max(6, len(labels) * 0.4)))
        y_pos = np.arange(len(labels))
        bars = plt.barh(y_pos, differences, color=colors)

        plt.yticks(y_pos, labels)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"{comparison_label} vs {baseline_label}\nTop Differences for {attribute_name} ({source})")
        plt.xlabel("Percentage Difference from Baseline")
        plt.gca().invert_yaxis()

        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + (0.5 if bar.get_width() > 0 else -0.5),
                bar.get_y() + bar.get_height() / 2,
                f"{differences[i]:+.1f}%",
                va='center',
                ha='left' if bar.get_width() > 0 else 'right',
                fontsize=8
            )

        # Save plot
        safe_attr = attribute_name.replace(" ", "_")
        safe_src = source.replace(" ", "_")
        safe_group = ''.join([w[0].upper() for w in comparison_label.split()])
        safe_base = ''.join([w[0].upper() for w in baseline_label.split()])
        fname = f"diff_{safe_group}{safe_base}_{safe_attr}_{safe_src}.png"
        plot_path = os.path.join(outdir, fname)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Saved plot: {plot_path}")

        # Optionally print metrics
        print(f"📊 Metrics for {attribute_name} ({source}):")
        for k, v in metrics.items():
            print(f"   {k}: {v}")
        print("-" * 50)



def plot_stacked_area_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str, 
    category_col: str,
    title: str = "Temporal Shifts Over Time",
    xlabel: str = "",
    ylabel: str = "",
    figsize: typing.Tuple[int, int] = (12, 8),
    output_path: typing.Optional[str] = None,
    color_palette: str = "tab20",
    alpha: float = 0.8,
    max_categories: int = 20,
    sort_by_total: bool = True
) -> plt.Figure:
    """
    Create a stacked area chart for temporal analysis.
    
    Args:
        data: DataFrame with your data
        x_col: Column name for x-axis values (e.g., "time", "date", "month")
        y_col: Column name for y-axis values (e.g., "count", "frequency", "value")
        category_col: Column name for categories (e.g., "topic", "function", "media_type")
        title: Plot title
        xlabel: Label for x-axis (if empty, uses x_col)
        ylabel: Label for y-axis (if empty, uses y_col)
        figsize: Figure size (width, height) in inches
        output_path: If provided, save the figure to this path
        color_palette: Matplotlib colormap name for category colors
        alpha: Transparency of the areas (0-1)
        max_categories: Maximum number of categories to display (top N by total)
        sort_by_total: Whether to sort categories by total value across time
    
    Returns:
        The Matplotlib Figure object.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use column names as default labels if not provided
    if not xlabel:
        xlabel = x_col
    if not ylabel:
        ylabel = y_col
    
    # Pivot data to wide format for plotting
    pivot_data = data.pivot(index=x_col, columns=category_col, values=y_col).fillna(0)
    
    # Limit categories if needed
    if len(pivot_data.columns) > max_categories:
        if sort_by_total:
            # Sort by total value across all time points
            category_totals = pivot_data.sum().sort_values(ascending=False)
            top_categories = category_totals.head(max_categories).index
            pivot_data = pivot_data[top_categories]
        else:
            # Just take first max_categories
            pivot_data = pivot_data.iloc[:, :max_categories]
    
    # Sort categories by total value for better visualization
    if sort_by_total:
        category_totals = pivot_data.sum().sort_values(ascending=False)
        pivot_data = pivot_data[category_totals.index]
    
    # Create stacked area plot
    colors = plt.cm.get_cmap(color_palette, len(pivot_data.columns))
    ax.stackplot(pivot_data.index, pivot_data.values.T, 
                 labels=pivot_data.columns, 
                 colors=colors(np.arange(len(pivot_data.columns))),
                 alpha=alpha)
    
    # Customize plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Rotate x-axis labels if they're long
    if len(str(pivot_data.index[0])) > 8:
        ax.tick_params(axis='x', rotation=45)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot if an output path is provided
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
    
    # Return the figure but prevent automatic display in Jupyter
    plt.close(fig)
    return fig


def plot_temporal_analysis(
    matrix: pd.DataFrame,
    title: str = "Temporal Shifts Over Time",
    figsize: typing.Tuple[int, int] = (14, 10),
    output_path: typing.Optional[str] = None,
    max_categories: int = 15
) -> plt.Figure:
    """
    Create a stacked area chart for temporal analysis of any annotation dimension.
    
    Args:
        matrix: DataFrame with categories as rows and time periods as columns
        title: Plot title
        figsize: Figure size (width, height) in inches
        output_path: If provided, save the figure to this path
        max_categories: Maximum number of categories to display (top N by total)
    
    Returns:
        The Matplotlib Figure object.
    """
    
    # ************* Customize the following *************
    
    # Convert matrix to long format for the general function
    long_data = matrix.reset_index().melt(
        id_vars='index',           # ← Customize if your time column isn't the index
        var_name='time',           # ← Customize the name of your time column
        value_name='value'         # ← Customize the name of your count/frequency column
    )
    long_data = long_data.rename(columns={'index': 'category'})  # ← Customize if your category column isn't the index
    
    # Filter to top categories by total value
    if len(long_data['category'].unique()) > max_categories:
        category_totals = long_data.groupby('category')['value'].sum().sort_values(ascending=False)
        top_categories = category_totals.head(max_categories).index
        long_data = long_data[long_data['category'].isin(top_categories)]
    
    # Create the stacked area chart
    fig = plot_stacked_area_chart(
        data=long_data,
        x_col='time',              # ← Must match the var_name from melt()
        y_col='value',             # ← Must match the value_name from melt()
        category_col='category',    # ← Must match the renamed column
        title=title,
        xlabel='Time Period',
        ylabel='Number of Occurrences',
        figsize=figsize,
        output_path=output_path,
        max_categories=max_categories
    )
    
    return fig

    # ************* Customize above *************


def plot_temporal_analysis_percentage(
    matrix: pd.DataFrame,
    title: str = "Temporal Shifts Over Time (Percentage)",
    figsize: typing.Tuple[int, int] = (14, 10),
    output_path: typing.Optional[str] = None,
    max_categories: int = 15
) -> plt.Figure:
    """
    Create a percentage-based stacked area chart for temporal analysis of any annotation dimension.
    
    Args:
        matrix: DataFrame with categories as rows and time periods as columns
        title: Plot title
        figsize: Figure size (width, height) in inches
        output_path: If provided, save the figure to this path
        max_categories: Maximum number of categories to display (top N by total)
    
    Returns:
        The Matplotlib Figure object.
    """
    # Convert matrix to long format for the general function
    long_data = matrix.reset_index().melt(
        id_vars='index', 
        var_name='time', 
        value_name='value'
    )
    long_data = long_data.rename(columns={'index': 'category'})
    
    # Filter to top categories by total value
    if len(long_data['category'].unique()) > max_categories:
        category_totals = long_data.groupby('category')['value'].sum().sort_values(ascending=False)
        top_categories = category_totals.head(max_categories).index
        long_data = long_data[long_data['category'].isin(top_categories)]
    
    # Create the percentage-based stacked area chart
    fig = plot_stacked_area_chart_percentage(
        data=long_data,
        x_col='time',
        y_col='value',
        category_col='category',
        title=title,
        xlabel='Time Period',
        ylabel='Percentage of Total',
        figsize=figsize,
        output_path=output_path,
        max_categories=max_categories
    )
    
    return fig


def plot_stacked_area_chart_percentage(
    data: pd.DataFrame,
    x_col: str,
    y_col: str, 
    category_col: str,
    title: str = "Temporal Shifts Over Time (Percentage)",
    xlabel: str = "",
    ylabel: str = "",
    figsize: typing.Tuple[int, int] = (12, 8),
    output_path: typing.Optional[str] = None,
    color_palette: str = "tab20",
    alpha: float = 0.8,
    max_categories: int = 20,
    sort_by_total: bool = True
) -> plt.Figure:
    """
    Create a percentage-based stacked area chart for temporal analysis.
    
    Args:
        data: DataFrame with your data
        x_col: Column name for x-axis values (e.g., "time", "date", "month")
        y_col: Column name for y-axis values (e.g., "count", "frequency", "value")
        category_col: Column name for categories (e.g., "topic", "function", "media_type")
        title: Plot title
        xlabel: Label for x-axis (if empty, uses x_col)
        ylabel: Label for y-axis (if empty, uses "Percentage")
        figsize: Figure size (width, height) in inches
        output_path: If provided, save the figure to this path
        color_palette: Matplotlib colormap name for category colors
        alpha: Transparency of the areas (0-1)
        max_categories: Maximum number of categories to display (top N by total)
        sort_by_total: Whether to sort categories by total value across time
    
    Returns:
        The Matplotlib Figure object.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use column names as default labels if not provided
    if not xlabel:
        xlabel = x_col
    if not ylabel:
        ylabel = "Percentage"
    
    # Pivot data to wide format for plotting
    pivot_data = data.pivot(index=x_col, columns=category_col, values=y_col).fillna(0)
    
    # Convert to percentages by dividing each column by the row sum
    pivot_data_percentage = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
    
    # Limit categories if needed
    if len(pivot_data_percentage.columns) > max_categories:
        if sort_by_total:
            # Sort by total percentage across all time points
            category_totals = pivot_data_percentage.sum().sort_values(ascending=False)
            top_categories = category_totals.head(max_categories).index
            pivot_data_percentage = pivot_data_percentage[top_categories]
        else:
            # Just take first max_categories
            pivot_data_percentage = pivot_data_percentage.iloc[:, :max_categories]
    
    # Sort categories by total percentage for better visualization
    if sort_by_total:
        category_totals = pivot_data_percentage.sum().sort_values(ascending=False)
        pivot_data_percentage = pivot_data_percentage[category_totals.index]
    
    # Create stacked area plot
    colors = plt.cm.get_cmap(color_palette, len(pivot_data_percentage.columns))
    ax.stackplot(pivot_data_percentage.index, pivot_data_percentage.values.T, 
                 labels=pivot_data_percentage.columns, 
                 colors=colors(np.arange(len(pivot_data_percentage.columns))),
                 alpha=alpha)
    
    # Customize plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set y-axis to show percentages from 0-100
    ax.set_ylim(0, 100)
    
    # Add percentage grid lines
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels if they're long
    if len(str(pivot_data_percentage.index[0])) > 8:
        ax.tick_params(axis='x', rotation=45)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot if an output path is provided
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
    
    # Return the figure but prevent automatic display in Jupyter
    plt.close(fig)
    return fig