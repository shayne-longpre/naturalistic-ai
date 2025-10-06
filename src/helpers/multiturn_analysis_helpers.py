# src/helpers/multiturn_analysis_helpers.py
"""
Helpers for multi-turn conversation analysis.

These utilities operate on *tidy* DataFrames. 
- Any “label-like” feature (topic, use, media type, etc.) is just a column name you pass in.
- Any “group” (model, cohort, time-slice) is just a column name you pass in.

Conventions for tidy inputs used below:
- id_col      : conversation/grouping id (e.g., "conversation_id", "conv_id")
- turn_col    : user-turn index (0,1,2,...) or message index (0,1,2,3,...) depending on your pipeline
- feature_col : categorical feature at each (user) turn (e.g., "label", "topic", "media_type")
- group_col   : group key for aggregation (e.g., "model", "split")

All functions return Pandas objects or simple Python structures, never figures.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd


# Generic token-length function #


def default_token_len_fn(text: str) -> int:
    """
    Very lightweight “token length” function based on whitespace splits.
    Replace with a real tokenizer if desired.
    """
    if not isinstance(text, str) or not text:
        return 0
    return len(text.split())


# Assistant-length stats (generic by group)  #


def compute_avg_response_length_by_group(
    msgs_df: pd.DataFrame,
    group_col: str,
    role_col: str = "role",
    text_col: str = "content",
    length_fn: Callable[[str], int] = default_token_len_fn,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Compute average assistant message length by group.

    Parameters
    ----------
    msgs_df : pd.DataFrame
        Must contain [group_col, role_col, text_col].
    group_col : str
        Group key (e.g., "model").
    role_col : str, default "role"
        Column containing role names (expects "assistant" rows to measure).
    text_col : str, default "content"
        Column containing assistant responses.
    length_fn : Callable[[str], int], default default_token_len_fn
        Function mapping text -> scalar length (words, tokens, chars, etc.).
    min_count : int, default 1
        Require at least this many assistant messages per group.

    Returns
    -------
    pd.DataFrame with columns:
        [group_col, "avg_assistant_length", "n_assistant_msgs"]
    """
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





# Conversation-level “change event” computations #


def _as_set_like(x):
    """
    Convert a value into a set for 'set_change' comparisons.
    Handles Python list/tuple/set and JSON-like strings.
    Falls back to {x} for scalars.
    """
    if isinstance(x, (list, tuple, set)):
        return set(x)
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            import json
            return set(json.loads(x))
        except Exception:
            return {x}
    return {x}


def compute_change_events_per_conversation(
    df: pd.DataFrame,
    id_col: str,
    turn_col: str,
    feature_col: str,
    step: int = 1,
    change_mode: str = "value_change",  # "value_change" | "set_change"
    distinct: bool = False,              # if True, count unique transition types per conversation
    return_transitions: bool = False,    # if True, include a list of transitions for each conversation
) -> pd.DataFrame:
    """
    Count how many times a conversation's feature changes across turns.

    Parameters
    ----------
    df : DataFrame
        Must contain [id_col, turn_col, feature_col].
    id_col : str
        Conversation identifier column (e.g., "conv_id").
    turn_col : str
        Turn index column (user-only turns if you're analyzing user prompts).
    feature_col : str
        The feature to track across turns (e.g., "label"). Can be scalar or list-like.
    step : int, default 1
        Compare (t) to (t+step). For user-only indices, 1 = adjacent user turns; 2 = skip-every-other.
    change_mode : {"value_change", "set_change"}
        - "value_change": compare values directly (a != b)
        - "set_change": treat values as list-like; compare set(a) != set(b)
    distinct : bool, default False
        If True, deduplicate transition *types* within each conversation.
        Example: A→B→A counts as 1 distinct type ("A→B") instead of 2 changes.
    return_transitions : bool, default False
        If True, also include a column 'transitions' with the list of (from, to) pairs considered.

    Returns
    -------
    pd.DataFrame
        Columns: [id_col, "change_events", "num_turns"] and optionally ["transitions"].
    """
    if change_mode not in {"value_change", "set_change"}:
        raise ValueError("change_mode must be 'value_change' or 'set_change'")

    out_rows: List[dict] = []

    # ensure required columns exist
    data = df[[id_col, turn_col, feature_col]].dropna(subset=[id_col, turn_col]).copy()

    for cid, sub in data.groupby(id_col, sort=False):
        sub = sub.sort_values(turn_col)
        vals = sub[feature_col].tolist()
        n = len(vals)

        if n <= step:
            row = {id_col: cid, "change_events": 0, "num_turns": n}
            if return_transitions:
                row["transitions"] = []
            out_rows.append(row)
            continue

        # Build the list of transitions (from, to) at the chosen step
        transitions: List[Tuple[Any, Any]] = []
        if change_mode == "set_change":
            for i in range(n - step):
                a, b = _as_set_like(vals[i]), _as_set_like(vals[i + step])
                if a != b:
                    transitions.append((tuple(sorted(a)), tuple(sorted(b))))
        else:  # "value_change"
            for i in range(n - step):
                a, b = vals[i], vals[i + step]
                if a != b:
                    transitions.append((a, b))

        change_count = len(set(transitions)) if distinct else len(transitions)

        row = {id_col: cid, "change_events": change_count, "num_turns": n}
        if return_transitions:
            row["transitions"] = transitions
        out_rows.append(row)

    return pd.DataFrame(out_rows)

def aggregate_change_events_by_group(
    changes_df: pd.DataFrame,
    id_to_group: pd.DataFrame,
    id_col: str,
    group_col: str,
    agg: str = "mean",   # "mean" or "median"
) -> pd.DataFrame:
    """
    Join conversation-level change_events with groups (e.g., model) and aggregate.

    Returns DataFrame with columns: [group_col, "avg_change_events"].
    """
    merged = changes_df.merge(id_to_group[[id_col, group_col]], on=id_col, how="left")
    if agg == "median":
        out = merged.groupby(group_col)["change_events"].median().reset_index(name="avg_change_events")
    else:
        out = merged.groupby(group_col)["change_events"].mean().reset_index(name="avg_change_events")
    return out


# Transition matrix + flows (Sankey)    #


def build_transition_matrix_from_tidy(
    df: pd.DataFrame,
    id_col: str,
    turn_col: str,
    feature_col: str,
    step: int = 1,
    labels_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a (labels x labels) transition **count** matrix from tidy rows over (t -> t+step).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [id_col, turn_col, feature_col].
    id_col : str
        Conversation id column.
    turn_col : str
        Turn index column (int).
    feature_col : str
        The label/feature column to transition on.
    step : int, default 1
        Transition step (t -> t+step). For user-only over mixed rows, use step=2.
    labels_order : Sequence[str], optional
        Custom order of matrix rows/cols. If None, alphabetical order of observed labels.

    Returns
    -------
    pd.DataFrame
        Square matrix (index=from, columns=to) with integer counts.
    """
    data = df[[id_col, turn_col, feature_col]].dropna().copy()

    if labels_order is None:
        labels_order = sorted(data[feature_col].dropna().unique())

    trans_counts = Counter()
    for _, sub in data.groupby(id_col, sort=False):
        sub = sub.sort_values(turn_col)
        seq = sub[feature_col].tolist()
        for i in range(len(seq) - step):
            trans_counts[(seq[i], seq[i + step])] += 1

    mat = pd.DataFrame(0, index=labels_order, columns=labels_order, dtype=int)
    for (a, b), c in trans_counts.items():
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] += c
    return mat


def build_flows_from_tidy(
    df: pd.DataFrame,
    id_col: str,
    turn_col: str,
    feature_col: str,
    sort_by_cols: Optional[Sequence[str]] = None,
) -> List[List[str]]:
    """
    Convert tidy rows to per-conversation sequences (flows) of feature values.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [id_col, turn_col, feature_col].
    id_col : str
        Conversation id.
    turn_col : str
        Turn index.
    feature_col : str
        Feature at each turn.
    sort_by_cols : Sequence[str], optional
        Additional sort columns for deterministic ordering (e.g., ["timestamp"]).

    Returns
    -------
    List[List[str]]
        List of feature sequences (one per conversation).
    """
    data = df.copy()
    if sort_by_cols:
        data = data.sort_values(list(sort_by_cols))
    flows: List[List[str]] = []
    for _, sub in data.groupby(id_col, sort=False):
        sub = sub.sort_values(turn_col)
        flows.append(sub[feature_col].tolist())
    return flows


def build_sankey_nodes_links(
    flows: List[List[str]],
    max_turns: int,
    stage_label_func: Optional[Callable[[int], str]] = None,  # e.g., lambda i: f"(t{i})"
    sort_labels_within_stage: bool = True,
) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Convert a list of label sequences to Sankey node/link arrays.

    Parameters
    ----------
    flows : List[List[str]]
        Sequences of feature/label values per conversation.
    max_turns : int
        Truncate each sequence to this many steps.
    stage_label_func : Callable[[int], str], optional
        Append a suffix to each stage label (e.g., "(t0)", "(t1)").
    sort_labels_within_stage : bool, default True
        Whether to alphabetically sort labels within each stage.

    Returns
    -------
    (node_labels, source_idx, target_idx, values)
        Arrays suitable for go.Sankey(node=..., link=...) construction.
    """
    from collections import Counter as _Counter

    stage_labels = [set() for _ in range(max_turns)]
    links_counter = _Counter()

    for seq in flows:
        seq = seq[:max_turns]
        if not seq:
            continue
        # register nodes
        for i, lbl in enumerate(seq):
            stage_labels[i].add(lbl)
        # register links
        for i in range(len(seq) - 1):
            links_counter[(i, seq[i], seq[i + 1])] += 1

    node_labels, node_index = [], {}
    for s in range(max_turns):
        ordered = sorted(stage_labels[s]) if sort_labels_within_stage else list(stage_labels[s])
        for lbl in ordered:
            node_index[(s, lbl)] = len(node_labels)
            suffix = f" {stage_label_func(s)}" if stage_label_func else ""
            node_labels.append(f"{lbl}{suffix}")

    src, tgt, val = [], [], []
    for (s, a, b), cnt in links_counter.items():
        if (s, a) in node_index and (s + 1, b) in node_index:
            src.append(node_index[(s, a)])
            tgt.append(node_index[(s + 1, b)])
            val.append(cnt)

    return node_labels, src, tgt, val


