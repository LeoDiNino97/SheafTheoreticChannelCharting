"""layers.py
Module containing wrappers for distance and loss computation for Siamese logic in neural networks
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceLayer(nn.Module):
    def __init__(self, distance_mode: str) -> None:

        super().__init__()
        assert distance_mode in ["euclidean", "cosine"], "Provide a valid distance mode"

        self.distance_mode = distance_mode

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        if self.distance_mode == "euclidean":
            return F.pairwise_distance(z1, z2)

        elif self.distance_mode == "cosine":
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            return 1 - F.cosine_similarity(z1, z2, dim=-1)

        else:
            return None


class LossLayer(nn.Module):
    def __init__(
        self,
        loss_mode: str,
        distance_mode: str,
        margin: float = 1.0,
    ) -> None:

        super().__init__()
        assert loss_mode in ["contrastive", "triplet"], "Provide a valid layer mode"
        assert distance_mode in ["euclidean", "cosine"], "Provide a valid distance mode"

        self.margin = margin
        self.loss_mode = loss_mode
        self.dist_layer = DistanceLayer(distance_mode=distance_mode)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        y: torch.Tensor,
        z3: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.loss_mode == "contrastive":
            assert y is not None
            y = y.float()

            d = self.dist_layer(z1, z2)
            dP = y * d.pow(2)
            dN = (1 - y) * torch.clamp(self.margin - d, min=0.0).pow(2)

            L = dP + dN

        elif self.loss_mode == "triplet":
            assert z3 is not None

            dAP = self.dist_layer(z1, z2)
            dAN = self.dist_layer(z1, z3)
            print(dAP, dAN)
            L = torch.clamp(dAP - dAN + self.margin, min=0.0)

        return L.mean()


class SiameseLayer(nn.Module):
    def __init__(
        self,
        loss_mode: str,
        distance_mode: str,
        # in_dim: int,
        margin: float,
    ) -> None:

        super().__init__()
        assert loss_mode in ["contrastive", "triplet"], "Provide a valid layer mode"
        assert distance_mode in ["euclidean", "cosine"], "Provide a valid distance mode"

        self.loss_mode = loss_mode
        self.distance_mode = distance_mode

        # self.in_dim = in_dim
        self.margin = margin

        self.loss_func = LossLayer(
            margin=margin,
            loss_mode=loss_mode,
            distance_mode=distance_mode,
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        y: torch.Tensor,
        z3: Optional[torch.Tensor] = None,
    ) -> None:
        return self.loss_func(z1=z1, z2=z2, z3=z3, y=y)
