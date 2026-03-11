import lightning as L
import torch
import torch.nn as nn


class NetworkOT(L.LightningModule):
    def __init__(
        self,
        agents: list[nn.Module],
        LR: float,
        L: torch.Tensor,
        B: torch.Tensor,
        n: int,
        lmb: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Agents list
        self.hparams['agents'] = nn.ModuleList(agents)

        # TODO
        # Network description
        self.hparams['edges'] = None

    def forward(self, batch):
        output = {}
        xA, xP, xN, _ = batch
        for agent in self.hparams.agents:
            xA_ = xA[
                :,
                agent.idx * self.hparams.n : (agent.idx + 1) * self.hparams.n,
            ]
            xP_ = xP[
                :,
                agent.idx * self.hparams.n : (agent.idx + 1) * self.hparams.n,
            ]
            xN_ = xN[
                :,
                agent.idx * self.hparams.n : (agent.idx + 1) * self.hparams.n,
            ]

            output[agent.idx] = agent(xA_, xP_, xN_)

        return output

    def _shared_eval(
        self, batch: list[torch.Tensor], batch_idx: int, prefix: str
    ):
        """A common step performed in the test and validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (output, total_loss) : tuple[dict[int, torch.Tensor], torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        main_loss = 0
        reg_loss = 0
        output = self(batch)

        for agent in self.hparams.agents:
            main_loss += agent.compute_loss(output[agent.idx])

        for i, j in self.hparams.edges:
            xi = (
                self.hparams.agents[i].M @ output[i][0].T
                + self.hparams.agents[i].b
            ) / self.hparams.agents[i].a
            xj = (
                self.hparams.agents[j].M @ output[j][0].T
                + self.hparams.agents[j].b
            ) / self.hparams.agents[j].a

            xi = xi.T
            xj = xj.T

            mask_i = output[i][0][:, 0] != 0
            mask_j = output[j][0][:, 0] != 0
            mask = mask_i & mask_j

            diff = xi - xj
            sq = torch.sum(diff**2, dim=1)

            reg_loss += torch.sum(sq * mask)

        total_loss = main_loss + self.hparams.lmb * reg_loss

        self.log(
            f'{prefix}/main_loss_epoch',
            main_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f'{prefix}/reg_loss_epoch', reg_loss, on_step=False, on_epoch=True
        )
        self.log(
            f'{prefix}/total_loss_epoch',
            total_loss,
            on_step=False,
            on_epoch=True,
        )

        return output, total_loss

    def training_step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """The training step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        _, loss = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='train',
        )
        return loss

    def test_step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
    ) -> None:
        """The test step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='test',
        )
        return None

    def validation_step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """The validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            output : dict[int, torch.Tensor]
                The output of the network.
        """
        output, _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='validation',
        )
        return output

    def predict_step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """The predict step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.

        Returns:
            dict[int, torch.Tensor]
                The output of the network.
        """
        return self(batch)

    def configure_optimizers(self) -> dict[str, object]:
        """Configure the optimizer used for training.

        Uses the AdamW optimizer with the learning rate defined in
        ``self.hparams.LR``.

        Returns:
            dict[str, object]: A dictionary containing the optimizer used by
            the training loop. The dictionary has the following key:

            - "optimizer": The instantiated AdamW optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.LR)
        return {
            'optimizer': optimizer,
        }
