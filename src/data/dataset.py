from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):

    def __init__(
        self,
        manifest_path: str | Path,
        class_to_idx: dict[str, int],
        transform: Optional[Callable] = None,
    ) -> None:

        self.manifest_path = Path(manifest_path)

        self.data = pd.read_csv(
            self.manifest_path
        )

        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(row["path"])

        image = Image.open(
            image_path
        ).convert("RGB")

        label = self.class_to_idx[
            str(row["class_code"])
        ]

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )