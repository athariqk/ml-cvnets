from typing import Optional

import torch
from cvnets.layers import LinearLayer
from cvnets.misc.init_utils import initialize_fc_layer
from torch import nn, Tensor

from cvnets.modules import BaseModule


class PhenotypeHead(BaseModule):
    def __init__(
            self,
            opts,
            in_channels: int,
            n_anchors: int,
            n_phenotypes: int,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        hidden_dim_width = n_anchors * n_phenotypes

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            LinearLayer(in_channels, hidden_dim_width),
            nn.ReLU(),
            LinearLayer(hidden_dim_width, n_phenotypes)
        )

        self.n_anchors = n_anchors
        self.n_phenotypes = n_phenotypes
        self.in_channel = in_channels

        self.reset_parameters()

    def __repr__(self) -> str:
        repr_str = "{}(in_channels={}, n_anchors={}, n_phenotypes={})".format(
            self.__class__.__name__,
            self.in_channel,
            self.n_anchors,
            self.n_phenotypes,
        )

        return repr_str

    def forward(self, x: Tensor) -> Tensor:
        pooled_features = self.pool(x)
        x = torch.flatten(pooled_features, 1)
        return self.regressor(x)
