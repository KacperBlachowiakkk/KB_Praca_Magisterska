from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import random



LIGHTING_FACTORS = {
    "original": 1.0,

    "dark_1": 0.70,
    "dark_2": 0.50,
    "dark_3": 0.30,

    "bright_1": 1.30,
    "bright_2": 1.60,
    "bright_3": 2.00,
}


LIGHTING_ORDER = [
    "original",
    "dark_1",
    "dark_2",
    "dark_3",
    "bright_1",
    "bright_2",
    "bright_3",
]


def apply_lighting_variant(
    image: Image.Image,
    variant: str,
) -> Image.Image:

    if variant not in LIGHTING_FACTORS:

        raise ValueError(
            f"Nieznany wariant oświetlenia: "
            f"{variant}"
        )

    image = image.convert("RGB")

    factor = LIGHTING_FACTORS[
        variant
    ]

    if factor == 1.0:
        return image

    enhancer = ImageEnhance.Brightness(
        image
    )

    return enhancer.enhance(
        factor
    )


class ControlledLightingDataset(
    Dataset
):

    def __init__(
        self,
        manifest_path: str | Path,
        class_to_idx: dict[str, int],
        lighting_variant: str,
        transform=None,
    ):

        self.manifest_path = Path(
            manifest_path
        )

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                f"Nie znaleziono manifestu: "
                f"{self.manifest_path}"
            )

        self.data = pd.read_csv(
            self.manifest_path
        )

        self.class_to_idx = (
            class_to_idx
        )

        self.lighting_variant = (
            lighting_variant
        )

        self.transform = transform

    def __len__(self):

        return len(
            self.data
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.data.iloc[
            index
        ]

        image_path = Path(
            row["path"]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:

            raise RuntimeError(
                f"Nie można odczytać obrazu: "
                f"{image_path}"
            ) from exc

        image = (
            apply_lighting_variant(
                image,
                self.lighting_variant,
            )
        )

        label = self.class_to_idx[
            str(
                row["class_code"]
            )
        ]

        if self.transform is not None:

            image = self.transform(
                image
            )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )
        


class RandomLightingAugmentation:

    def __init__(
        self,
        brightness_range=(0.5, 1.3),
        contrast_range=(0.8, 1.2),
        probability=0.8,
    ):

        self.brightness_range = (
            brightness_range
        )

        self.contrast_range = (
            contrast_range
        )

        self.probability = probability


    def __call__(
        self,
        image,
    ):

        image = image.convert(
            "RGB"
        )


        if (
            random.random()
            > self.probability
        ):

            return image


        brightness_factor = (
            random.uniform(
                self.brightness_range[0],
                self.brightness_range[1],
            )
        )


        contrast_factor = (
            random.uniform(
                self.contrast_range[0],
                self.contrast_range[1],
            )
        )


        image = (
            ImageEnhance.Brightness(
                image
            ).enhance(
                brightness_factor
            )
        )


        image = (
            ImageEnhance.Contrast(
                image
            ).enhance(
                contrast_factor
            )
        )


        return image