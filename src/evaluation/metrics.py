import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@torch.no_grad()
def predict(
    model,
    loader,
    device,
):

    model.eval()

    y_true = []
    y_pred = []

    start_time = time.perf_counter()

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        predictions = (
            logits
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        y_true.extend(
            targets.numpy().tolist()
        )

        y_pred.extend(
            predictions.tolist()
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        elapsed,
    )


def evaluate_predictions(
    y_true,
    y_pred,
):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(
            precision
        ),
        "macro_recall": float(
            recall
        ),
        "macro_f1": float(
            f1
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ),
        "classification_report": (
            classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            )
        ),
    }