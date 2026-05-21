"""
Label utilities for EEG seizure detection.

Functions
---------
get_majority_consensus : Determine the majority label from a list of expert
                         annotations, returning None when there is a tie.
build_label_map        : Build an {eeg_id -> label} dictionary from a metadata
                         DataFrame using majority-consensus voting.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd


def get_majority_consensus(consensus_list: list[str]) -> Optional[str]:
    """
    Return the most frequent label in *consensus_list*, or ``None`` on a tie.

    Parameters
    ----------
    consensus_list : Sequence of expert label strings, e.g.
                     ``["Seizure", "Seizure", "LPD"]``.

    Returns
    -------
    The majority label string, or ``None`` if no single label has a strict
    majority (i.e. two or more labels share the highest count).

    Examples
    --------
    >>> get_majority_consensus(["Seizure", "Seizure", "LPD"])
    'Seizure'

    >>> get_majority_consensus(["Seizure", "Seizure", "LPD", "LPD"])
    None

    >>> get_majority_consensus(["LPD"])
    'LPD'
    """
    if not consensus_list:
        return None

    counts = Counter(consensus_list).most_common(2)

    if len(counts) == 1:
        return counts[0][0]

    if counts[0][1] > counts[1][1]:
        return counts[0][0]

    return None  # tie


def build_label_map(
    df: pd.DataFrame,
    id_column: str = "eeg_id",
    label_column: str = "expert_consensus",
) -> dict[str, Optional[str]]:
    """
    Aggregate per-row expert labels to one majority label per EEG recording.

    Parameters
    ----------
    df           : DataFrame with one row per expert annotation.
    id_column    : Column containing EEG recording IDs.
    label_column : Column containing individual expert label strings.

    Returns
    -------
    Dictionary mapping each EEG ID (as a string) to its majority label, or
    ``None`` where no consensus could be reached.
    """
    grouped = df.groupby(id_column)[label_column].agg(list)
    return {
        str(eeg_id): get_majority_consensus(labels)
        for eeg_id, labels in grouped.items()
    }
