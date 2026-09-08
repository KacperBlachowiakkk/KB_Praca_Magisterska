from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)

import torch.nn as nn


def create_mobilenet_v3(
    num_classes: int,
    pretrained: bool = True,
):
    weights = (
        MobileNet_V3_Large_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_large(
        weights=weights
    )

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes,
    )

    return model


def freeze_mobilenet_v3_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model


def unfreeze_mobilenet_v3_last_block(model):
    for parameter in model.features[-3:].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model