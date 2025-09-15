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
import plotly.graph_objects as go
import plotly.express as px

sys.path.append("./")

from src.helpers.constants import FUNCTION_ANNOTATION_LABEL_ABBREVIATIONS
from src.helpers.dataset_comparison import aggregate_counts_by_category, group_counts_into_parent_categories

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



def make_tree_plot(category_counts: typing.Dict[str, int], parent_dict: typing.Dict[str:str], save_path:str=None, split_lines_display:bool=False, print_counts:bool=False):
    """
    Create a treemap visualization using Plotly to show hierarchical category distributions.
    Args:
        category_counts (dict): Dictionary with category names as keys and their counts as values.
        parent_dict (dict): Dictionary mapping parent categories to lists of their child categories.
        save_path (str, optional): If provided, save the treemap image to this path
        split_lines_display (bool): If True, split long category labels into multiple lines for better display.
        print_counts (bool): If True, print out the counts for each label for debugging purposes
    Returns:
        values (list): List of counts corresponding to each label.
        parents (list): List of parent labels corresponding to each label.
        labels (list): List of all labels (both parents and children).
        fig (plotly.graph_objects.Figure): The Plotly treemap figure object.

    Example dictionaries as input:
    category_counts = {
        'Apples': 150,
        'Bananas': 100,
        'Broccoli': 80,
        }
    parent_dict = {
        'Fruits': ['Apples', 'Bananas'],
        'Vegetables': ['Broccoli'],
        }

    """
    labels = []
    parents = []
    values = []

    # If desired, replace any commas in child labels with <br> for better display in Plotly. This breaks up a long label into multiple lines. 
    def format_label(label):
        label = label.replace(',', '')
        label = label.replace('/', '')
        return label.replace(' ', '<br>')#.replace('&', '<br>&')

    # Prepare formatted parent_dict for label mapping (only format children)
    if split_lines_display:
        formatted_parent_dict = {parent: [format_label(child) for child in children] for parent, children in parent_dict.items()}

    # Use formatted labels for all further processing
    labels = []
    parents = []
    values = []

    # Build a reverse mapping from subcategory to parent (with formatted child labels)
    child_to_parent = {}
    for parent, children in formatted_parent_dict.items():
        for child in children:
            child_to_parent[child] = parent

    # Find which parents have at least one child in category_counts (to avoid plotting empy boxes)
    parents_with_children = set()
    for cat in category_counts:
        formatted_cat = format_label(cat)
        parent = child_to_parent.get(formatted_cat)
        if parent is not None:
            parents_with_children.add(parent)

    # Add only those parent nodes to the labels, parents, and values lists
    for parent in parents_with_children:
        labels.append(parent)
        parents.append('')
        values.append(0)

    # Add all categories in category_counts (children)
    for cat, count in category_counts.items():
        formatted_cat = format_label(cat)
        parent = child_to_parent.get(formatted_cat)
        if parent is not None:
            labels.append(formatted_cat)
            parents.append(parent)
            values.append(count)
        else:
            print(f"Warning: Category '{cat}' has no parent in parent_dict. Skipping it.") 
    
    #For debugging, print out the counts for each label.
    if print_counts:
        for lbl, val in zip(labels, values):
            print(f"{lbl}: {val}")

    # Create the treemap figure
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        textinfo="label+percent entry",
    ))
    
    fig.update_traces(textfont_size=20)
    fig.update_layout(width=2000, height=1400)
  
    if save_path is not None:
        fig.write_image(save_path)

    return values, parents, labels, fig


def make_spider_plot(data_to_compare, parent_dict = None, title = "", save_path = None):
    """
    This function takes in a dictionary of the category counts from 1+ datasets, whose values are plotted in a spider plot and named according to the key. 
    If desired, a parent_dict can be applied to aggregate the categories into more abstract groups. 
    If desired, a title can be specified. 

    Example Input: 
    data_to_compare = {
        'dataset1': {'apples': 19, 'bananas: 42, 'broccoli': 10}, 
        'dataset2': {'apples': 27, 'bananas: 21, 'broccoli: 12 }, 
    }
    parent_dict = {
        'fruits': ['apples', 'bananas'], 
        'vegetables': ['broccoli'],
    }
    title = "Dataset 1 vs Dataset 2, Amount of Fruits and Vegetables "

    """

    categories =list(next(iter(data_to_compare.values())).keys())
    
    if parent_dict is not None:
        grouped_data_to_compare = {
            name: group_counts_into_parent_categories(count_dict, parent_dict=parent_dict)
            for name, count_dict in data_to_compare.items()
        }
        # Use the keys from the first dataset in grouped_data_to_compare for categories
        categories = list(next(iter(grouped_data_to_compare.values())).keys())
        data_to_compare = grouped_data_to_compare


    # Remove categories where all datasets have 0 occurrences
    categories_to_keep = [
        cat for cat in categories
        if any(data_to_compare[ds].get(cat, 0) > 0 for ds in data_to_compare)
    ]
    categories = categories_to_keep

    # Prepare values for each dataset
    values_to_plot = {}

    for dataset_name, agg_wc in data_to_compare.items():
        print(f"Processing dataset: {dataset_name}")
        values = [agg_wc.get(cat, 0) for cat in categories]
        if len(values) >= 1:
            values += [values[0]]  # Close the loop for the radar plot
        values_to_plot[dataset_name] = values

    # Make scatterpolar 
    fig = go.Figure(
        data=[ 
            go.Scatterpolar(
                r=v,
                theta=categories,
                fill='toself',
                name=k
            ) for k, v in values_to_plot.items()
        ]
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        title=f"{title}",
        showlegend=True,
        width=1500,
        height=800
    )
    fig.show()
    if save_path is not None: 
        fig.write_image(save_path)

    return fig

def make_heatmap(comparisons:dict[str:dict[str:int]], baseline:dict[str:int], title:str, save_path:str = None, fig_width:int = 800):
    """
    This function produces a heatmap used plot the difference of several datasets to a single baseline. Each row represents a dataset and each column represents a category. 
    The value plotted in the heatmap is the numerical difference between the comparion datasets' value for that category and the baseline's value for that category. 

    Inputs:  
    Comparisons is a dictionary structured as name => dictionary of categories and counts. Each item in this dictionary represents a row of the heatmap. 
    The baseline is a single dictionary of categories to counts. The baseline's keys (categories) are used throughout. 
    Each row will be titled by the name in the comparisons dict automatically, and the figure is titled with the title string provided. 
    
    """
    all_keys = set(baseline.keys()) # these will each be a column
    for comp_dict in comparisons.values():
        all_keys.update(comp_dict.keys())

    # Ensure all dicts have all keys, fill missing with 0
    baseline = {k: baseline.get(k, 0) for k in sorted(all_keys)}
    for name in comparisons:
        comparisons[name] = {k: comparisons[name].get(k, 0) for k in sorted(all_keys)}

    # Create a matrix where each row is the difference between a comparison dict and the baselin
    z = []
    names = []
    for name, comp in comparisons.items():
        diff = {k: (comp[k] - baseline[k] ) for k in baseline.keys()}
        z.append([diff[k] for k in sorted(diff.keys())])
        names.append(name)


    # Colorscale for range -2000 to 2000, with 0 as grey
    colorscale = [
        [0.0, "darkblue"],        # -2000
        [0.4, "lightsteelblue"], # -1000
        [0.5, "lightgrey"],       # 0
        [0.9, "salmon"],         # +1000
        [1.0, "red"]              # +2000
    ]

    fig = px.imshow(
        z,
        text_auto=True,
        x=sorted(baseline.keys()),
        y=names,
        color_continuous_scale=colorscale,
        zmin=-1000,
        zmax=1000
    )

    fig.update_layout(coloraxis_colorscale=colorscale)
    fig.update_xaxes(tickangle=0)
    fig.update_xaxes(
        ticktext=[
            label.replace(' and ', ' and<br>').replace('&', '&<br>') if 'and' in label else label.replace('&', '&<br>')
            for label in fig.layout.xaxis.ticktext or fig.data[0].x
        ],
        tickvals=list(range(len(fig.data[0].x)))
    )
    fig.update_xaxes(title_font=dict(size=24))
    fig.update_yaxes(title_font=dict(size=24))
    fig.update_layout(
        width=fig_width,
        height=600
    )
    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_layout(xaxis_side="top")
    if save_path:
        fig.write_image(save_path)

    fig.show()
