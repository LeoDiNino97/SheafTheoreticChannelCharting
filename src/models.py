"""layers.py
Module containing classes implemented as in "Siamese Neural Networks for Wireless Positioning and Channel Charting"
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import *


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 2,
        num_hidden_layers: int = 6,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_hidden_layers = num_hidden_layers
        self.act = F.relu

        self.layers = nn.ModuleList()

        dim = in_dim * 2

        for i in range(self.num_hidden_layers - 1):
            self.layers.append(nn.Linear(dim, dim // 2))
            dim = dim // 2

        self.layers.append(nn.Linear(dim, self.out_dim))

    def forward(self, x: torch.Tensor) -> None:
        for i in range(self.num_hidden_layers - 1):
            x = self.act(self.layers[i](x))
        return self.layers[-1](x)


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 256,
        num_hidden_layers: int = 6,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_hidden_layers = num_hidden_layers
        self.act = F.relu

        self.layers = nn.ModuleList()

        dim = in_dim * 2

        for i in range(self.num_hidden_layers - 1):
            self.layers.append(nn.Linear(dim, dim * 2))
            dim = dim * 2

        self.layers.append(nn.Linear(dim, self.out_dim))

    def forward(self, x: torch.Tensor) -> None:
        for i in range(self.num_hidden_layers - 1):
            x = self.act(self.layers[i](x))
        return self.layers[-1](x)


class SiameseNeuralNetwork(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        distance_mode: str,
        loss_mode: str,
        in_dim: int = 256,
        out_dim: int = None,
        num_hidden_layers: int = 6,
        margin: float = 1.0,
    ) -> None:
        super().__init__()

        assert arch in ["Encoder", "Autoencoder"], "Provide a valid architecture flag"
        assert distance_mode in ["euclidean", "cosine"], "Provide a valid distance mode"
        assert loss_mode in ["contrastive", "triplet"], "Provide a valid layer mode"

        self.arch = arch
        self.distance_mode = distance_mode
        self.loss_mode = loss_mode

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.margin = margin

        if arch == "Autoencoder":
            assert in_dim == out_dim, (
                "Unsolvable design choices! Provide a combination for in and out dimension coherent with the chosen architecture"
            )
            self.encoder = Encoder(
                in_dim=in_dim, out_dim=2, num_hidden_layers=num_hidden_layers
            )

            self.decoder = Decoder(
                in_dim=2, out_dim=in_dim, num_hidden_layers=num_hidden_layers
            )

            self.siamese = SiameseLayer(
                loss_mode=loss_mode,
                distance_mode=distance_mode,
                in_dim=in_dim,
                margin=margin,
            )

        else:
            self.encoder = Encoder(
                in_dim=in_dim, out_dim=out_dim, num_hidden_layers=num_hidden_layers
            )

            self.decoder = nn.Identity()

            self.siamese = SiameseLayer(
                loss_mode=loss_mode,
                distance_mode=distance_mode,
                in_dim=out_dim,
                margin=margin,
            )

    def training_step(self, batch, batch_idx) -> None:

        xA, xP, xN, y = batch
        embA = self.decoder(self.encoder(xA))
        embP = self.decoder(self.encoder(xP))
        if xN is not None:
            embN = self.decoder(self.encoder(xN))
        else:
            embN = None
        train_loss = self.siamese(embA, embP, embN, y)
        self.log("Training loss:", train_loss)

    def validation_step(self, batch, batch_idx) -> None:

        xA, xP, xN, y = batch
        embA = self.decoder(self.encoder(xA))
        embP = self.decoder(self.encoder(xP))
        if xN is not None:
            embN = self.decoder(self.encoder(xN))
        else:
            embN = None
        val_loss = self.siamese(embA, embP, embN, y)
        self.log("Validation loss", val_loss)

    def test_step(self, batch, batch_idx) -> None:

        xA, xP, xN, y = batch
        embA = self.decoder(self.encoder(xA))
        embP = self.decoder(self.encoder(xP))
        if xN is not None:
            embN = self.decoder(self.encoder(xN))
        else:
            embN = None

        test_loss = self.siamese(embA, embP, embN, y)
        self.log("Test loss:", test_loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
