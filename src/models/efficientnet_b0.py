from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

import torch.nn as nn


def create_efficientnet_b0(
    num_classes: int,
    pretrained: bool = True,
):
    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = efficientnet_b0(
        weights=weights
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    return model


def freeze_efficientnet_b0_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model


def unfreeze_efficientnet_b0_last_block(model):
    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model