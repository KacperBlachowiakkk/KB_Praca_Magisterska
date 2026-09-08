from __future__ import annotations
import time
from pathlib import Path
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch=None,
    total_epochs=None,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    description = (
        f"Epoka {epoch}/{total_epochs}"
        if epoch is not None
        else "Trening"
    )

    progress_bar = tqdm(
        loader,
        desc=description,
        leave=True,
    )

    for images, targets in progress_bar:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            logits.argmax(dim=1)
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

        progress_bar.set_postfix(
            loss=f"{running_loss / max(total, 1):.4f}",
            acc=f"{correct / max(total, 1):.4f}",
        )

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            targets,
        )

        predictions = (
            logits.argmax(dim=1)
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

        y_true.extend(
            targets.cpu().numpy().tolist()
        )

        y_pred.extend(
            predictions.cpu().numpy().tolist()
        )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
    }


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    checkpoint_path=None,
):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_macro_f1 = float("-inf")
    best_state = None

    start_time = time.perf_counter()

    epoch_progress = tqdm(
        range(1, epochs + 1),
        desc="Postęp treningu",
    )

    for epoch in epoch_progress:

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        validation_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(
            train_stats["loss"]
        )

        history["train_accuracy"].append(
            train_stats["accuracy"]
        )

        history["val_loss"].append(
            validation_stats["loss"]
        )

        history["val_accuracy"].append(
            validation_stats["accuracy"]
        )

        history["val_macro_f1"].append(
            validation_stats["macro_f1"]
        )

        epoch_progress.set_postfix(
            train_loss=f"{train_stats['loss']:.4f}",
            train_acc=f"{train_stats['accuracy']:.4f}",
            val_acc=f"{validation_stats['accuracy']:.4f}",
            val_f1=f"{validation_stats['macro_f1']:.4f}",
        )

        if (
            validation_stats["macro_f1"]
            > best_macro_f1
        ):
            best_macro_f1 = (
                validation_stats["macro_f1"]
            )

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            if checkpoint_path is not None:

                checkpoint_path = Path(
                    checkpoint_path
                )

                checkpoint_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                torch.save(
                    best_state,
                    checkpoint_path,
                )

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    history["training_time_seconds"] = (
        time.perf_counter()
        - start_time
    )

    history["best_val_macro_f1"] = (
        best_macro_f1
    )

    return history