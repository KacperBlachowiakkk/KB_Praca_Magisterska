from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)

import torch.nn as nn


def create_resnet50(
    num_classes: int,
    pretrained: bool = True,
):
    weights = (
        ResNet50_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet50(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def freeze_resnet50_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


def unfreeze_resnet50_last_block(model):
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model