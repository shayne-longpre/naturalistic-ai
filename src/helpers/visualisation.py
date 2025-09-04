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


# user and multiturn visuals helper functions

from typing import Optional, Sequence, Dict, List, Tuple, Union, Callable
from matplotlib.ticker import PercentFormatter

# 1) 100% stacked horizontal bars by group (e.g., per model)
def plot_stacked_100h_by_group(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    weight_col: Optional[str] = None,                 # if None uses counts; else sum of weights
    exclude_labels: Optional[Sequence[str]] = None,   # e.g., ["First request"]
    order_groups_by: Union[str, Sequence[str]] = "size",  # "size" | "alpha" | custom sequence
    labels_order: Optional[Sequence[str]] = None,     # custom order of stacked segments
    palette: Optional[Sequence] = None,
    title: str = "",
    xlabel: str = "Proportion",
    ylabel: str = "",
    figsize: Tuple[int, int] = (12, 6),
    legend_ncol: Optional[int] = None,
    annotate_min_prop: float = 0.05,                  # show % label if >= threshold
    bar_height: float = 0.8,
    percent_formatter: Optional[PercentFormatter] = PercentFormatter(1.0),
    dpi: int = 150,
    output_path: Optional[str] = None,
    show: bool = True, 
) -> plt.Figure:
    data = df.copy()
    if exclude_labels:
        data = data[~data[label_col].isin(exclude_labels)]

    # counts or weights
    if weight_col is None:
        dist = (
            data.groupby([group_col, label_col]).size()
                .rename("count").reset_index()
        )
    else:
        dist = (
            data.groupby([group_col, label_col])[weight_col]
                .sum().rename("count").reset_index()
        )
    # proportions
    dist["proportion"] = dist.groupby(group_col)["count"].transform(lambda x: x / x.sum())

    # pivot to wide
    pivot_df = dist.pivot(index=group_col, columns=label_col, values="proportion").fillna(0)

    # order groups
    if isinstance(order_groups_by, (list, tuple)):
        order_groups = [g for g in order_groups_by if g in pivot_df.index]
        pivot_df = pivot_df.loc[order_groups]
    elif order_groups_by == "size":
        sizes = data.groupby(group_col).size().sort_values(ascending=False)
        pivot_df = pivot_df.loc[sizes.index.intersection(pivot_df.index)]
    elif order_groups_by == "alpha":
        pivot_df = pivot_df.sort_index()

    # order labels (stack segments)
    if labels_order is None:
        labels_order = list(sorted(pivot_df.columns))

    colors = palette or sns.color_palette("tab10", n_colors=len(labels_order))

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(pivot_df))

    # draw stacks
    for i, lbl in enumerate(labels_order):
        if lbl not in pivot_df.columns:
            vals = np.zeros(len(pivot_df))
        else:
            vals = pivot_df[lbl].values
        bars = ax.barh(
            pivot_df.index, vals, left=bottom, color=colors[i % len(colors)],
            edgecolor="white", linewidth=0.5, label=lbl, height=bar_height
        )
        # annotate inside bar if large enough
        for bar, v in zip(bars, vals):
            if v >= annotate_min_prop:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + bar.get_height() / 2
                ax.text(x, y, f"{v*100:.0f}%", va="center", ha="center",
                        fontsize=9, color="white")
        bottom += vals

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if percent_formatter:
        ax.xaxis.set_major_formatter(percent_formatter)

    legend_cols = legend_ncol or min(4, max(1, len(labels_order)))
    ax.legend(title="Label", bbox_to_anchor=(0.5, -0.15), loc="upper center",
              ncol=legend_cols, frameon=False)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

# Faceted line plot of label proportions over user turns, split by group (e.g., model)
def plot_faceted_label_trends_by_group(
    df: pd.DataFrame,
    group_col: str,
    turn_col: str,
    label_col: str,
    weight_col: Optional[str] = None,           # None: count rows; else sum weights
    top_k_labels: Optional[int] = None,         # collapse others to "Other"
    exclude_labels: Optional[Sequence[str]] = None,
    turn_cutoff: Optional[int] = None,          
    labels_order: Optional[Sequence[str]] = None,
    palette: Optional[Sequence] = None,
    title: str = "",
    xlabel: str = "User Turn Index",
    ylabel: str = "Proportion",
    col_wrap: int = 3,
    height: float = 3.0,
    aspect: float = 1.3,
    xticks_step: Optional[int] = None,          # e.g., 2 for even-turn spacing labels
    dpi: int = 150,
    output_path: Optional[str] = None,
    show: bool = True, 
) -> plt.Figure:
    data = df.copy()

    if exclude_labels:
        data = data[~data[label_col].isin(exclude_labels)]

    if top_k_labels is not None:
        counts = data[label_col].value_counts()
        keep = set(counts.index[:top_k_labels])
        data[label_col] = np.where(data[label_col].isin(keep), data[label_col], "Other")

    if turn_cutoff is not None:
        data = data[data[turn_col] < turn_cutoff]

    # compute counts/proportions per (group, turn, label)
    if weight_col is None:
        grp = (
            data.groupby([group_col, turn_col, label_col])
                .size().rename("count").reset_index()
        )
    else:
        grp = (
            data.groupby([group_col, turn_col, label_col])[weight_col]
                .sum().rename("count").reset_index()
        )
    grp["total"] = grp.groupby([group_col, turn_col])["count"].transform("sum")
    grp["proportion"] = grp["count"] / grp["total"]

    # label order + palette
    labels_sorted = labels_order or sorted(grp[label_col].unique())
    pal = palette or sns.color_palette("tab10", n_colors=len(labels_sorted))

    g = sns.FacetGrid(
        data=grp, col=group_col, col_wrap=col_wrap, height=height, aspect=aspect,
        sharey=True, sharex=True, hue=label_col, palette=pal, hue_order=labels_sorted
    )
    g.map_dataframe(sns.lineplot, x=turn_col, y="proportion", marker="o")
    g.add_legend(title="Label", bbox_to_anchor=(1.02, 0.5), loc="center left")
    g.set_axis_labels(xlabel, ylabel)
    g.set_titles("{col_name}")

    # x ticks
    if turn_cutoff is not None or xticks_step is not None:
        xmax = int(grp[turn_col].max()) if turn_cutoff is None else turn_cutoff - 1
        ticks = range(0, xmax + 1, xticks_step or 1)
        for ax in g.axes.flatten():
            ax.set_xticks(ticks)

    plt.subplots_adjust(bottom=0.15, right=0.82, top=0.9)
    g.fig.suptitle(title, y=0.98, fontsize=13)

    if output_path:
        g.fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(g.fig)
    return g.fig


# Overall label trends over user turns (single axes)
def plot_label_trends_overall(
    df: pd.DataFrame,
    turn_col: str,
    label_col: str,
    weight_col: Optional[str] = None,
    top_k_labels: Optional[int] = None,
    exclude_labels: Optional[Sequence[str]] = None,
    turn_cutoff: Optional[int] = None,
    labels_order: Optional[Sequence[str]] = None,
    palette: Optional[Sequence] = None,
    title: str = "",
    xlabel: str = "User Turn Index",
    ylabel: str = "Proportion",
    figsize: Tuple[int, int] = (9, 5),
    xticks_step: Optional[int] = None,
    dpi: int = 150,
    output_path: Optional[str] = None,
    show: bool = True, 
) -> plt.Figure:
    data = df.copy()

    if exclude_labels:
        data = data[~data[label_col].isin(exclude_labels)]

    if top_k_labels is not None:
        counts = data[label_col].value_counts()
        keep = set(counts.index[:top_k_labels])
        data[label_col] = np.where(data[label_col].isin(keep), data[label_col], "Other")

    if turn_cutoff is not None:
        data = data[data[turn_col] < turn_cutoff]

    if weight_col is None:
        grp = data.groupby([turn_col, label_col]).size().rename("count").reset_index()
    else:
        grp = data.groupby([turn_col, label_col])[weight_col].sum().rename("count").reset_index()

    grp["total"] = grp.groupby(turn_col)["count"].transform("sum")
    grp["proportion"] = grp["count"] / grp["total"]

    labels_sorted = labels_order or sorted(grp[label_col].unique())
    pal = palette or sns.color_palette("tab10", n_colors=len(labels_sorted))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(labels_sorted):
        sub = grp[grp[label_col] == lbl]
        ax.plot(sub[turn_col], sub["proportion"], marker="o",
                label=lbl, color=pal[i % len(pal)])

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(title="Label", bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)

    if turn_cutoff is not None or xticks_step is not None:
        xmax = int(grp[turn_col].max()) if turn_cutoff is None else turn_cutoff - 1
        ticks = range(0, xmax + 1, xticks_step or 1)
        ax.set_xticks(ticks)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# Sankey (overall or per group) — fully parameterized
def sankey_from_flows(
    flows: List[List[str]],
    max_user_turns: int,
    stage_label_func: Optional[Callable[[int], str]] = None,  # e.g., lambda s: f"(t{2*s})"
    sort_labels_within_stage: bool = True,
) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Convert list of label sequences to Sankey node/link arrays.
    flows: [[l0,l1,l2,...], ...] (user-turn labels only)
    Returns: (node_labels, source_idx, target_idx, values)
    """
    from collections import Counter as _Counter

    stage_labels = [set() for _ in range(max_user_turns)]
    links_counter = _Counter()

    for seq in flows:
        seq = seq[:max_user_turns]
        if len(seq) < 2:
            # still register single nodes so they appear
            for i, lbl in enumerate(seq):
                stage_labels[i].add(lbl)
            continue
        for i, lbl in enumerate(seq):
            stage_labels[i].add(lbl)
        for i in range(len(seq) - 1):
            links_counter[(i, seq[i], seq[i+1])] += 1

    node_labels, node_index = [], {}
    for s in range(max_user_turns):
        ordered = sorted(stage_labels[s]) if sort_labels_within_stage else list(stage_labels[s])
        for lbl in ordered:
            node_index[(s, lbl)] = len(node_labels)
            suffix = f" {stage_label_func(s)}" if stage_label_func else ""
            node_labels.append(f"{lbl}{suffix}")

    src, tgt, val = [], [], []
    for (s, a, b), cnt in links_counter.items():
        if (s, a) in node_index and (s+1, b) in node_index:
            src.append(node_index[(s, a)])
            tgt.append(node_index[(s+1, b)])
            val.append(cnt)
    return node_labels, src, tgt, val


def plot_sankey(
    flows: List[List[str]],
    max_user_turns: int = 4,
    stage_label_func: Optional[Callable[[int], str]] = lambda s: f"(t{2*s})",
    title: str = "",
    width: int = 980,
    height: int = 560,
    font_size: int = 12,
    save_html: Optional[str] = None,
    show: bool = True, 


):
    import plotly.graph_objects as go
    labels, s, t, v = sankey_from_flows(
        flows, max_user_turns=max_user_turns, stage_label_func=stage_label_func
    )
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=18, thickness=18, line=dict(color="black", width=0.5), label=labels),
        link=dict(source=s, target=t, value=v)
    )])
    fig.update_layout(title=title, font_size=font_size, width=width, height=height)
    if save_html:
        os.makedirs(os.path.dirname(save_html), exist_ok=True)
        fig.write_html(save_html)
    if show:
        fig.show()
    return fig

# Build flows from tidy per-turn DF
def build_flows_from_tidy(
    df: pd.DataFrame,
    id_col: str,
    turn_col: str,
    label_col: str,
    sort_by_cols: Optional[Sequence[str]] = None,  
) -> List[List[str]]:
    data = df.copy()
    if sort_by_cols:
        data = data.sort_values(list(sort_by_cols))
    flows = []
    for _, sub in data.groupby(id_col, sort=False):
        sub = sub.sort_values(turn_col)
        flows.append(sub[label_col].tolist())
    return flows








def build_transition_matrix_from_tidy(
    df: pd.DataFrame,
    id_col: str,          # conversation/grouping id (e.g., conversation_id)
    turn_col: str,        # user-turn index (0,1,2,... for user-only turns; or any integer index)
    label_col: str,       # categorical label at each user turn
    even_only: bool = False,   # if True: compute t -> t+2 (useful for raw mixed turns), else adjacent turns
    labels_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a (labels x labels) transition count matrix from tidy rows.
    Assumes df contains ONLY user turns for easiest interpretation (or set even_only=True).
    """
    data = df[[id_col, turn_col, label_col]].dropna().copy()

    if labels_order is None:
        labels_order = sorted(data[label_col].dropna().unique())

    trans_counts = Counter()
    for _, sub in data.groupby(id_col, sort=False):
        sub = sub.sort_values(turn_col)
        seq = sub[label_col].tolist()
        if even_only:
            for i in range(len(seq) - 2):
                trans_counts[(seq[i], seq[i+2])] += 1
        else:
            for i in range(len(seq) - 1):
                trans_counts[(seq[i], seq[i+1])] += 1

    mat = pd.DataFrame(0, index=labels_order, columns=labels_order, dtype=int)
    for (a, b), c in trans_counts.items():
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] += c
    return mat



def default_token_len_fn(text: str) -> int:
    """Very simple length function (tokens ≈ whitespace-split). Override if needed."""
    if not isinstance(text, str) or not text:
        return 0
    return len(text.split())

def plot_avg_assistant_length_by_group(
    msgs_df: pd.DataFrame,
    group_col: str = "model",
    role_col: str = "role",
    text_col: str = "content",
    length_fn: Optional[Callable[[str], int]] = None,   # default: word count
    min_count: int = 1,                                  # min # assistant msgs per group
    order: str = "desc",                                 # "asc" | "desc" | "alpha"
    title: str = "Average Assistant Response Length by Group",
    xlabel: str = "Group",
    ylabel: str = "Avg # Tokens",
    rotation: int = 45,
    figsize: Tuple[int, int] = (9, 4),
    dpi: int = 150,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compute and plot average assistant response length by group (e.g., by model).

    Args:
        msgs_df: DataFrame with columns [group_col, role_col, text_col]
        group_col: Column to group by (e.g., "model")
        role_col: Role column (expects "assistant" rows to measure)
        text_col: Text column containing assistant responses
        length_fn: Function mapping text -> length. Defaults to word count.
                   Replace with your tokenizer if desired.
        min_count: Require at least this many assistant messages per group
        order: How to order groups: "asc" | "desc" | "alpha"
        title, xlabel, ylabel: Plot labels
        rotation: x tick label rotation
        figsize, dpi: Figure sizing
        output_path: If provided, saves PNG to this path
        show: If True, calls plt.show()

    Returns:
        Matplotlib Figure
    """
    if length_fn is None:
        length_fn = lambda s: len(str(s).split())

    # --- Filter to assistant messages and compute lengths ---
    df = msgs_df[[group_col, role_col, text_col]].dropna(subset=[group_col, role_col, text_col]).copy()
    df = df[df[role_col] == "assistant"].copy()
    if df.empty:
        # Return empty figure gracefully
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No assistant messages to plot.", ha="center", va="center")
        ax.axis("off")
        if output_path: fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        if show: plt.show()
        plt.close(fig)
        return fig

    df["msg_len"] = df[text_col].apply(length_fn)

    agg = (
        df.groupby(group_col)["msg_len"]
          .agg(["mean", "count"])
          .rename(columns={"mean": "avg_assistant_length", "count": "n_assistant_msgs"})
          .reset_index()
    )

    # Apply min_count filter
    agg = agg[agg["n_assistant_msgs"] >= min_count].copy()
    if agg.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No groups with ≥ {min_count} assistant messages.", ha="center", va="center")
        ax.axis("off")
        if output_path: fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        if show: plt.show()
        plt.close(fig)
        return fig

    # Order groups
    if order == "alpha":
        agg = agg.sort_values(group_col)
    elif order == "asc":
        agg = agg.sort_values("avg_assistant_length", ascending=True)
    else:
        agg = agg.sort_values("avg_assistant_length", ascending=False)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=agg, x=group_col, y="avg_assistant_length", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)

    # Annotate bars with values
    for p in ax.patches:
        v = p.get_height()
        ax.text(p.get_x() + p.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compute_topic_shifts_per_conversation(
    df: pd.DataFrame,
    id_col: str,           # conversation_id
    turn_col: str,         # user-turn index
    label_col: str,        # label at each user turn
    even_only: bool = False,
) -> pd.DataFrame:
    """
    Count how many times a conversation changes label across user turns.
    Returns a DataFrame with columns: [id_col, "topic_shifts", "num_user_turns"].
    """
    rows = []
    for cid, sub in df[[id_col, turn_col, label_col]].dropna().groupby(id_col, sort=False):
        sub = sub.sort_values(turn_col)
        seq = sub[label_col].tolist()
        if not seq:
            continue
        if even_only:
            comp_pairs = zip(seq, seq[2:])
        else:
            comp_pairs = zip(seq, seq[1:])
        shifts = sum(1 for a, b in comp_pairs if a != b)
        rows.append({id_col: cid, "topic_shifts": shifts, "num_user_turns": len(seq)})
    return pd.DataFrame(rows)


def aggregate_topic_shifts_by_group(
    shifts_df: pd.DataFrame,
    id_to_group: pd.DataFrame,   # two cols: [id_col, group_col], one row per conversation
    id_col: str,
    group_col: str,
    agg: str = "mean",           # "mean" or "median"
) -> pd.DataFrame:
    """
    Join conversation-level shifts with group label, then aggregate per group.
    Returns: DataFrame with columns [group_col, "avg_topic_shifts"] (or median).
    """
    merged = shifts_df.merge(id_to_group[[id_col, group_col]], on=id_col, how="left")
    if agg == "median":
        out = merged.groupby(group_col)["topic_shifts"].median().reset_index(name="avg_topic_shifts")
    else:
        out = merged.groupby(group_col)["topic_shifts"].mean().reset_index(name="avg_topic_shifts")
    return out



def plot_group_metric_bar(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    order: str = "desc",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rotation: int = 45,
    figsize: Tuple[int, int] = (9, 4),
    dpi: int = 150,
    output_path: Optional[str] = None,
    show: bool = True, 
) -> plt.Figure:
    data = df[[group_col, metric_col]].dropna().copy()
    if order == "alpha":
        data = data.sort_values(group_col)
    elif order == "asc":
        data = data.sort_values(metric_col, ascending=True)
    else:
        data = data.sort_values(metric_col, ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x=group_col, y=metric_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def compute_avg_response_length_by_group(
    msgs_df: pd.DataFrame,
    group_col: str,
    role_col: str = "role",
    text_col: str = "content",
    length_fn: Callable[[str], int] = default_token_len_fn,
    min_count: int = 1,
) -> pd.DataFrame:
    df = msgs_df[[group_col, role_col, text_col]].dropna().copy()
    df = df[df[role_col] == "assistant"].copy()
    if df.empty:
        return pd.DataFrame(columns=[group_col, "avg_assistant_length", "n_assistant_msgs"])

    df["msg_len"] = df[text_col].apply(length_fn)
    agg = (
        df.groupby(group_col)["msg_len"]
          .agg(["mean", "count"])
          .rename(columns={"mean": "avg_assistant_length", "count": "n_assistant_msgs"})
          .reset_index()
    )
    if min_count > 1:
        agg = agg[agg["n_assistant_msgs"] >= min_count]
    return agg


# --- Plot: average assistant length by group ---
def plot_avg_assistant_length_by_group(
    msgs_df: pd.DataFrame,
    group_col: str,
    role_col: str = "role",
    text_col: str = "content",
    length_fn: Callable[[str], int] = default_token_len_fn,
    order: str = "desc",
    title: str = "Average Assistant Response Length by Group",
    xlabel: str = "",
    ylabel: str = "Avg # Tokens",
    rotation: int = 45,
    figsize: Tuple[int, int] = (9, 4),
    dpi: int = 150,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    stats = compute_avg_response_length_by_group(
        msgs_df, group_col=group_col, role_col=role_col, text_col=text_col, length_fn=length_fn
    )
    if stats.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No assistant messages found")
        return fig

    # Order
    if order == "alpha":
        stats = stats.sort_values(group_col)
    elif order == "asc":
        stats = stats.sort_values("avg_assistant_length", ascending=True)
    else:
        stats = stats.sort_values("avg_assistant_length", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.barplot(data=stats, x=group_col, y="avg_assistant_length", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_transition_heatmap_matrix(
    matrix: pd.DataFrame,
    normalize: bool = True,            # row-normalize to proportions
    title: str = "Label Transition Matrix",
    xlabel: str = "To",
    ylabel: str = "From",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    vmin: Optional[float] = None,      # set to 0 for proportions
    vmax: Optional[float] = None,      # set to 1 for proportions
    annot: bool = True,
    annot_fontsize: int = 8,
    square: bool = False,
    dpi: int = 150,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    plot_mat = matrix.copy()
    fmt = "d"
    if normalize:
        plot_mat = plot_mat.div(plot_mat.sum(axis=1), axis=0).fillna(0)
        fmt = ".2f"
        if vmin is None: vmin = 0.0
        if vmax is None: vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(
        plot_mat, annot=annot, fmt=fmt, cmap=cmap, ax=ax,
        vmin=vmin, vmax=vmax, square=square, cbar=True,
        cbar_kws=dict(shrink=0.8)
    )
    if annot and annot_fontsize:
        for t in hm.texts:
            t.set_fontsize(annot_fontsize)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
