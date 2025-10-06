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
from matplotlib.ticker import PercentFormatter


sys.path.append("./")

from src.helpers.constants import FUNCTION_ANNOTATION_LABEL_ABBREVIATIONS
from src.helpers.multiturn_analysis_helpers import (
    build_transition_matrix_from_tidy,
    build_flows_from_tidy,
    build_sankey_nodes_links,
    compute_avg_response_length_by_group,  
)

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


# =====  USER & MULTI-TURN VISUALS (PLOTTING ONLY)  =======


def plot_stacked_100h_by_group(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    weight_col: typing.Optional[str] = None,
    exclude_labels: typing.Optional[typing.Sequence[str]] = None,   # e.g., ["First request"]
    order_groups_by: typing.Union[str, typing.Sequence[str]] = "size",  # "size" | "alpha" | custom sequence
    labels_order: typing.Optional[typing.Sequence[str]] = None,     # custom order of stacked segments
    palette: typing.Optional[typing.Sequence] = None,
    title: str = "",
    xlabel: str = "Proportion",
    ylabel: str = "",
    figsize: typing.Tuple[int, int] = (12, 6),
    legend_ncol: typing.Optional[int] = None,
    annotate_min_prop: float = 0.05,                  # show % label if >= threshold
    bar_height: float = 0.8,
    percent_formatter: typing.Optional[PercentFormatter] = PercentFormatter(1.0),
    dpi: int = 150,
    output_path: typing.Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a 100% stacked horizontal bar chart of label distribution within each group.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy dataframe containing at least [group_col, label_col] and optionally [weight_col].
    group_col : str
        Column indicating groups (e.g., model name).
    label_col : str
        Categorical column to stack (e.g., prompt type).
    weight_col : Optional[str], default None
        If provided, sums weights per (group,label) instead of counting rows.
    exclude_labels : Optional[Sequence[str]], default None
        Labels to drop before computing proportions.
    order_groups_by : {"size","alpha", sequence}, default "size"
        - "size": order groups by total count/weight descending
        - "alpha": alphabetical by group label
        - sequence: explicit ordering list of group names
    labels_order : Optional[Sequence[str]], default None
        Explicit ordering for stacked segment labels. Defaults to sorted columns.
    palette : Optional[Sequence], default None
        List of colors. Defaults to seaborn "tab10" with sufficient colors.
    title, xlabel, ylabel : str
        Plot text elements.
    figsize : (int,int), default (12,6)
        Figure size in inches.
    legend_ncol : Optional[int], default None
        Number of legend columns. Defaults to min(4, len(labels)).
    annotate_min_prop : float, default 0.05
        Minimum proportion threshold for drawing in-bar % labels.
    bar_height : float, default 0.8
        Height of each horizontal bar.
    percent_formatter : Optional[PercentFormatter], default PercentFormatter(1.0)
        Formatter for x-axis (0–1 -> 0–100%). Pass None to disable percent ticks.
    dpi : int, default 150
        Save figure DPI if output_path is provided.
    output_path : Optional[str], default None
        If provided, saves figure to this path.
    show : bool, default True
        If True, calls plt.show(); else closes the figure (useful for batch jobs).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    data = df.copy()
    if exclude_labels:
        data = data[~data[label_col].isin(exclude_labels)]

    # ---- counts or weights per (group, label) ----
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

    # ---- proportions within each group ----
    dist["proportion"] = dist.groupby(group_col)["count"].transform(lambda x: x / x.sum())

    # ---- wide pivot: rows=groups, cols=labels ----
    pivot_df = dist.pivot(index=group_col, columns=label_col, values="proportion").fillna(0)

    # ---- order groups ----
    if isinstance(order_groups_by, (list, tuple)):
        # keep only requested groups that exist
        order_groups = [g for g in order_groups_by if g in pivot_df.index]
        pivot_df = pivot_df.loc[order_groups]
    elif order_groups_by == "size":
        if weight_col is None:
            sizes = data.groupby(group_col).size().sort_values(ascending=False)
        else:
            sizes = (
                data.groupby(group_col)[weight_col]
                    .sum().sort_values(ascending=False)
            )
        pivot_df = pivot_df.loc[sizes.index.intersection(pivot_df.index)]
    elif order_groups_by == "alpha":
        pivot_df = pivot_df.sort_index()

    # ---- order labels (stack order) ----
    if labels_order is None:
        labels_order = list(sorted(pivot_df.columns))

    # ---- colors ----
    colors = palette or sns.color_palette("tab10", n_colors=max(1, len(labels_order)))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(pivot_df), dtype=float)

    for i, lbl in enumerate(labels_order):
        vals = pivot_df[lbl].values if lbl in pivot_df.columns else np.zeros(len(pivot_df))
        bars = ax.barh(
            pivot_df.index, vals, left=bottom, color=colors[i % len(colors)],
            edgecolor="white", linewidth=0.5, label=lbl, height=bar_height
        )
        # annotate inside bar if large enough
        for bar, v in zip(bars, vals):
            if v >= annotate_min_prop and bar.get_width() > 0:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + bar.get_height() / 2
                ax.text(x, y, f"{v*100:.0f}%", va="center", ha="center",
                        fontsize=9, color="white")
        bottom += vals

    # ---- axes/legend ----
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if percent_formatter:
        ax.xaxis.set_major_formatter(percent_formatter)

    legend_cols = legend_ncol or min(4, max(1, len(labels_order)))
    ax.legend(title="Label", bbox_to_anchor=(0.5, -0.15), loc="upper center",
              ncol=legend_cols, frameon=False)

    plt.tight_layout()

    # ---- save/show ----
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_line_trends_over_turns_overall(
    df: pd.DataFrame,
    turn_col: str,
    feature_col: str,
    weight_col: typing.Optional[str] = None,
    top_k_values: typing.Optional[int] = None,
    exclude_values: typing.Optional[typing.Sequence[str]] = None,
    turn_cutoff: typing.Optional[int] = None,
    values_order: typing.Optional[typing.Sequence[str]] = None,
    palette: typing.Optional[typing.Sequence] = None,
    title: str = "",
    xlabel: str = "Turn",
    ylabel: str = "Proportion",
    figsize: typing.Tuple[int, int] = (9, 5),
    xticks_step: typing.Optional[int] = None,
    dpi: int = 150,
    output_path: typing.Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Line plot of feature proportions over turns (single axis).
    Expects df with [turn_col, feature_col] (+ optional weight_col).
    """
    data = df.copy()
    if exclude_values:
        data = data[~data[feature_col].isin(exclude_values)]
    if top_k_values is not None:
        counts = data[feature_col].value_counts()
        keep = set(counts.index[:top_k_values])
        data[feature_col] = np.where(data[feature_col].isin(keep), data[feature_col], "Other")
    if turn_cutoff is not None:
        data = data[data[turn_col] < turn_cutoff]

    if weight_col is None:
        grp = data.groupby([turn_col, feature_col]).size().rename("count").reset_index()
    else:
        grp = data.groupby([turn_col, feature_col])[weight_col].sum().rename("count").reset_index()

    grp["total"] = grp.groupby(turn_col)["count"].transform("sum")
    grp["proportion"] = grp["count"] / grp["total"]

    values_sorted = values_order or sorted(grp[feature_col].unique())
    pal = palette or sns.color_palette("tab10", n_colors=len(values_sorted))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i, val in enumerate(values_sorted):
        sub = grp[grp[feature_col] == val]
        ax.plot(sub[turn_col], sub["proportion"], marker="o", label=val, color=pal[i % len(pal)])

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(title=feature_col, bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)

    if turn_cutoff is not None or xticks_step is not None:
        xmax = int(grp[turn_col].max()) if turn_cutoff is None else turn_cutoff - 1
        ax.set_xticks(range(0, xmax + 1, xticks_step or 1))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_line_trends_over_turns_by_group(
    df: pd.DataFrame,
    group_col: str,
    turn_col: str,
    feature_col: str,
    weight_col: typing.Optional[str] = None,
    top_k_values: typing.Optional[int] = None,
    exclude_values: typing.Optional[typing.Sequence[str]] = None,
    turn_cutoff: typing.Optional[int] = None,
    values_order: typing.Optional[typing.Sequence[str]] = None,
    palette: typing.Optional[typing.Sequence] = None,
    title: str = "",
    xlabel: str = "Turn",
    ylabel: str = "Proportion",
    col_wrap: int = 3,
    height: float = 3.0,
    aspect: float = 1.3,
    xticks_step: typing.Optional[int] = None,
    dpi: int = 150,
    output_path: typing.Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Faceted line plots of feature proportions over turns, split by group.
    Expects df with [group_col, turn_col, feature_col] (+ optional weight_col).
    """
    data = df.copy()
    if exclude_values:
        data = data[~data[feature_col].isin(exclude_values)]
    if top_k_values is not None:
        counts = data[feature_col].value_counts()
        keep = set(counts.index[:top_k_values])
        data[feature_col] = np.where(data[feature_col].isin(keep), data[feature_col], "Other")
    if turn_cutoff is not None:
        data = data[data[turn_col] < turn_cutoff]

    # compute counts/proportions per (group, turn, feature)
    if weight_col is None:
        grp = (
            data.groupby([group_col, turn_col, feature_col])
                .size().rename("count").reset_index()
        )
    else:
        grp = (
            data.groupby([group_col, turn_col, feature_col])[weight_col]
                .sum().rename("count").reset_index()
        )
    grp["total"] = grp.groupby([group_col, turn_col])["count"].transform("sum")
    grp["proportion"] = grp["count"] / grp["total"]

    values_sorted = values_order or sorted(grp[feature_col].unique())
    pal = palette or sns.color_palette("tab10", n_colors=len(values_sorted))

    g = sns.FacetGrid(
        data=grp, col=group_col, col_wrap=col_wrap, height=height, aspect=aspect,
        sharey=True, sharex=True, hue=feature_col, palette=pal, hue_order=values_sorted
    )
    g.map_dataframe(sns.lineplot, x=turn_col, y="proportion", marker="o")
    g.add_legend(title=feature_col, bbox_to_anchor=(1.02, 0.5), loc="center left")
    g.set_axis_labels(xlabel, ylabel)
    g.set_titles("{col_name}")

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


def plot_transition_heatmap_matrix(
    matrix: pd.DataFrame,
    normalize: bool = True,
    title: str = "Transition Matrix",
    xlabel: str = "To",
    ylabel: str = "From",
    figsize: typing.Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    vmin: typing.Optional[float] = None,
    vmax: typing.Optional[float] = None,
    annot: bool = True,
    annot_fontsize: int = 8,
    square: bool = False,
    dpi: int = 150,
    output_path: typing.Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a single transition matrix (counts or row-normalized proportions).
    """
    plot_mat = matrix.copy()
    fmt = "d"
    if normalize:
        plot_mat = plot_mat.div(plot_mat.sum(axis=1), axis=0).fillna(0)
        fmt = ".2f"
        if vmin is None: vmin = 0.0
        if vmax is None: vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    hm = sns.heatmap(
        plot_mat, annot=annot, fmt=fmt, cmap=cmap, ax=ax,
        vmin=vmin, vmax=vmax, square=square, cbar=True, cbar_kws=dict(shrink=0.8)
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


def plot_transition_heatmaps_by_group(
    df: pd.DataFrame,
    group_col: str,
    id_col: str,
    turn_col: str,
    feature_col: str,
    step: int = 1,
    normalize: bool = True,
    labels_order: typing.Optional[typing.Sequence[str]] = None,
    title_prefix: str = "Transitions —",
    cols: int = 3,
    base_cell_size: typing.Tuple[float, float] = (5.0, 4.0),  # scales up overall size so panels aren't tiny
    cmap: str = "Blues",
    annot: bool = True,
    annot_fontsize: int = 8,
    square: bool = False,
    dpi: int = 150,
    output_path: typing.Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Build and plot a grid of transition heatmaps—one per group (e.g., per model).
    Figure size is dynamically scaled by the grid size.
    """
    groups = list(df[group_col].dropna().unique())
    if not groups:
        raise ValueError(f"No groups found for '{group_col}'")

    rows = (len(groups) + cols - 1) // cols
    fig_w = base_cell_size[0] * cols
    fig_h = base_cell_size[1] * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.array(axes).reshape(-1)

    for i, g in enumerate(groups):
        sub = df[df[group_col] == g]
        mat = build_transition_matrix_from_tidy(
            sub, id_col=id_col, turn_col=turn_col, feature_col=feature_col,
            step=step, labels_order=labels_order
        )

        # Prepare plot matrix
        plot_mat = mat.copy()
        fmt = "d"; vmin = vmax = None
        if normalize:
            plot_mat = plot_mat.div(plot_mat.sum(axis=1), axis=0).fillna(0.0)
            fmt = ".2f"; vmin, vmax = 0.0, 1.0

        ax = axes[i]
        hm = sns.heatmap(
            plot_mat, ax=ax, annot=annot, fmt=fmt, cmap=cmap, square=square,
            vmin=vmin, vmax=vmax, cbar=(i == 0), cbar_kws=dict(shrink=0.8 if i == 0 else 1.0)
        )
        if annot and annot_fontsize:
            for t in hm.texts:
                t.set_fontsize(annot_fontsize)

        ax.set_title(f"{title_prefix} {g}", fontsize=11)
        ax.set_xlabel("To")
        ax.set_ylabel("From")

    # hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_avg_assistant_length_by_group(
    msgs_df: pd.DataFrame,
    group_col: str = "model",
    role_col: str = "role",
    text_col: str = "content",
    length_fn: typing.Optional[typing.Callable[[str], int]] = None,   # default: word count
    min_count: int = 1,
    order: str = "desc",                                 # "asc" | "desc" | "alpha"
    title: str = "Average Assistant Response Length by Group",
    xlabel: str = "Group",
    ylabel: str = "Avg # Tokens",
    rotation: int = 45,
    figsize: typing.Tuple[int, int] = (9, 4),
    dpi: int = 150,
    output_path: typing.Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compute + plot average assistant message length by group (e.g., by model).
    Uses `compute_avg_response_length_by_group` from helpers.
    """
    if length_fn is None:
        length_fn = lambda s: len(str(s).split())

    stats = compute_avg_response_length_by_group(
        msgs_df=msgs_df,
        group_col=group_col,
        role_col=role_col,
        text_col=text_col,
        length_fn=length_fn,
        min_count=min_count,
    )

    if stats.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No assistant messages to plot.", ha="center", va="center")
        ax.axis("off")
        if output_path: fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        if show: plt.show()
        plt.close(fig)
        return fig

    if order == "alpha":
        stats = stats.sort_values(group_col)
    elif order == "asc":
        stats = stats.sort_values("avg_assistant_length", ascending=True)
    else:
        stats = stats.sort_values("avg_assistant_length", ascending=False)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.barplot(data=stats, x=group_col, y="avg_assistant_length", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)

    # Value labels
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


def plot_sankey(
    flows: typing.List[typing.List[str]],
    max_turns: int = 4,
    stage_label_func: typing.Optional[typing.Callable[[int], str]] = lambda s: f"(t{s})",
    title: str = "",
    width: int = 980,
    height: int = 560,
    font_size: int = 12,
    save_html: typing.Optional[str] = None,
    show: bool = True,
):
    """
    Plot a Sankey diagram given per-conversation flows of feature labels.

    Build flows via `build_flows_from_tidy`, and nodes/links via `build_sankey_nodes_links`.
    """
    import plotly.graph_objects as go

    labels, s, t, v = build_sankey_nodes_links(
        flows=flows, max_turns=max_turns, stage_label_func=stage_label_func
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


def plot_group_metric_bar(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    order: str = "desc",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rotation: int = 45,
    figsize: typing.Tuple[int, int] = (9, 4),
    dpi: int = 150,
    output_path: typing.Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Simple bar chart for [group_col -> metric_col].
    """
    data = df[[group_col, metric_col]].dropna().copy()
    if order == "alpha":
        data = data.sort_values(group_col)
    elif order == "asc":
        data = data.sort_values(metric_col, ascending=True)
    else:
        data = data.sort_values(metric_col, ascending=False)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.barplot(data=data, x=group_col, y=metric_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel(ylabel or metric_col)
    ax.tick_params(axis="x", rotation=rotation)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig