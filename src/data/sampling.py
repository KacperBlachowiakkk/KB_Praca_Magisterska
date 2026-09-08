from __future__ import annotations
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(
    manifest_path: str | Path,
    class_to_idx: dict[str, int],
) -> torch.Tensor:

    dataframe = pd.read_csv(manifest_path)

    counts = (
        dataframe["class_code"]
        .astype(str)
        .value_counts()
    )

    weights = torch.ones(
        len(class_to_idx),
        dtype=torch.float32,
    )

    for class_code, class_index in class_to_idx.items():
        if class_code in counts:
            weights[class_index] = (
                1.0 / float(counts[class_code])
            )

    weights = weights / weights.mean()

    return weights


def build_weighted_sampler(
    manifest_path: str | Path,
) -> WeightedRandomSampler:

    dataframe = pd.read_csv(manifest_path)

    labels = (
        dataframe["class_code"]
        .astype(str)
        .tolist()
    )

    class_counts = Counter(labels)

    sample_weights = torch.tensor(
        [
            1.0 / class_counts[class_code]
            for class_code in labels
        ],
        dtype=torch.double,
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )