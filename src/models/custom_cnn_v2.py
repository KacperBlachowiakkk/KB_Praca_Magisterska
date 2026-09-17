from __future__ import annotations
import torch
import torch.nn as nn


class CustomCNNV2(nn.Module):

    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),


            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )


    def forward(
        self,
        x: torch.Tensor,
    ):

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


def create_custom_cnn_v2(
    num_classes: int,
):

    return CustomCNNV2(
        num_classes
    )