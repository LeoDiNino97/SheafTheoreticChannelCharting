"""layers.py
Module containing classes implemented as in
"Siamese Neural Networks for Wireless Positioning and Channel Charting"
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import SiameseLayer


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_hidden_layers = num_hidden_layers
        self.act = F.relu

        self.layers = nn.ModuleList()

        dim = in_dim

        for _ in range(self.num_hidden_layers - 1):
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

        dim = in_dim

        for _ in range(self.num_hidden_layers - 1):
            self.layers.append(nn.Linear(dim, dim * 2))
            dim = dim * 2

        self.layers.append(nn.Linear(dim, self.out_dim))

    def forward(self, x: torch.Tensor) -> None:
        for i in range(self.num_hidden_layers - 1):
            x = self.act(self.layers[i](x))
        return self.layers[-1](x)


class SiameseNN(L.LightningModule):
    def __init__(
        self,
        distance_mode: str,
        loss_mode: str,
        in_dim: int,
        autoenc: bool = False,
        out_dim: int = 2,
        num_hidden_layers: int = 6,
        margin: float = 1.0,
        lr: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__()

        assert distance_mode in ['euclidean', 'cosine'], (
            'Provide a valid distance mode'
        )
        assert loss_mode in ['contrastive', 'triplet'], (
            'Provide a valid layer mode'
        )

        self.autoenc = autoenc
        self.distance_mode = distance_mode
        self.loss_mode = loss_mode

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.margin = margin
        self.lr = lr

        if autoenc:
            assert in_dim == out_dim, (
                'Unsolvable design choices! Provide a combination for in'
                'and out dimension coherent with the chosen architecture'
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
                # in_dim=in_dim,
                margin=margin,
            )

        else:
            self.encoder = Encoder(
                in_dim=in_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
            )

            self.decoder = nn.Identity()

            self.siamese = SiameseLayer(
                loss_mode=loss_mode,
                distance_mode=distance_mode,
                # in_dim=out_dim,
                margin=margin,
            )

    def forward(self, batch) -> None:
        """
        Wrapper for the forward pass in all lightning steps
        """
        xA, xP, xN, y = batch
        embA = self.decoder(self.encoder(xA))
        embP = self.decoder(self.encoder(xP))
        embN = self.decoder(self.encoder(xN)) if xN is not None else None
        return embA, embP, embN, y

    def training_step(self, batch, batch_idx) -> None:
        embA, embP, embN, y = self.forward(batch)
        train_loss = self.siamese(z1=embA, z2=embP, z3=embN, y=y)
        self.log('Training loss:', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx) -> None:
        embA, embP, embN, y = self.forward(batch)
        val_loss = self.siamese(z1=embA, z2=embP, z3=embN, y=y)
        self.log('Validation loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx) -> None:
        embA, embP, embN, y = self.forward(batch)
        test_loss = self.siamese(z1=embA, z2=embP, z3=embN, y=y)
        self.log('Test loss:', test_loss)
        return test_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
