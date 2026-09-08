from __future__ import annotations
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms


def identity(
    image: Image.Image,
) -> Image.Image:

    return image.convert("RGB")


def gamma_correction(
    image: Image.Image,
    gamma: float = 1.5,
) -> Image.Image:

    if gamma <= 0:
        raise ValueError(
            "Wartość gamma musi być większa od 0."
        )

    image = image.convert("RGB")

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    corrected = np.power(
        array,
        1.0 / gamma,
    )

    corrected = np.clip(
        corrected * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        corrected
    )


def histogram_equalization(
    image: Image.Image,
) -> Image.Image:

    image = image.convert("RGB")

    array = np.asarray(
        image
    )

    ycrcb = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2YCrCb,
    )

    y_channel = ycrcb[
        :,
        :,
        0,
    ]

    y_equalized = cv2.equalizeHist(
        y_channel
    )

    ycrcb[
        :,
        :,
        0,
    ] = y_equalized

    result = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2RGB,
    )

    return Image.fromarray(
        result
    )


def clahe(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:

    image = image.convert("RGB")

    array = np.asarray(
        image
    )

    lab = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2LAB,
    )

    l_channel, a_channel, b_channel = (
        cv2.split(lab)
    )

    clahe_operator = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    l_enhanced = (
        clahe_operator.apply(
            l_channel
        )
    )

    enhanced_lab = cv2.merge(
        [
            l_enhanced,
            a_channel,
            b_channel,
        ]
    )

    result = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB,
    )

    return Image.fromarray(
        result
    )


def build_preprocessor(
    method: str,
    gamma: float = 1.5,
):

    if method == "original":

        return identity

    if method == "gamma":

        return lambda image: gamma_correction(
            image,
            gamma=gamma,
        )

    if method == "histogram_equalization":

        return histogram_equalization

    if method == "clahe":

        return clahe

    raise ValueError(
        f"Nieznana metoda preprocessingu: "
        f"{method}"
    )


def build_preprocessing_eval_transform(
    method: str,
    image_size: int = 224,
    gamma: float = 1.5,
):

    preprocessor = (
        build_preprocessor(
            method,
            gamma=gamma,
        )
    )

    return transforms.Compose(
        [
            transforms.Lambda(
                preprocessor
            ),

            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=(
                    0.485,
                    0.456,
                    0.406,
                ),
                std=(
                    0.229,
                    0.224,
                    0.225,
                ),
            ),
        ]
    )


def build_preprocessing_train_transform(
    method: str,
    image_size: int = 224,
    gamma: float = 1.5,
):

    preprocessor = (
        build_preprocessor(
            method,
            gamma=gamma,
        )
    )

    return transforms.Compose(
        [
            transforms.Lambda(
                preprocessor
            ),

            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.RandomRotation(
                10
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=(
                    0.485,
                    0.456,
                    0.406,
                ),
                std=(
                    0.229,
                    0.224,
                    0.225,
                ),
            ),
        ]
    )