import lightning as L
import torch
import torch.nn as nn


class NetworkAgent(L.LightningModule):
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

    def on_train_epoch_end(self):
        dataloader = self.trainer.datamodule.train_dataloader()

        self.eval()

        with torch.no_grad():
            # Reset local aggregators of cross-covariance
            for agent in self.hparams.agents:
                agent.reset_epoch_statistics()

            # Compute and aggregate embeddings
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)

                output = self(batch)

                for i, j in self.hparams.edges:
                    self.hparams.agents[i].accumulate_statistics(
                        output[i][0], output[j][0], self.hparams.agents[j].R
                    )
                    self.hparams.agents[j].accumulate_statistics(
                        output[j][0], output[i][0], self.hparams.agents[i].R
                    )

            # Perform reference alignment
            for agent in self.hparams.agents:
                agent.update_reference_frame()

        self.train()

        return None

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
        """A common step performend in the test and validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (y_hat, loss) : tuple[torch.Tensor, torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        R = torch.zeros_like(self.hparams.L)
        loss = 0
        output = self(batch)
        E = torch.cat(
            [output[agent.idx] for agent in self.hparams.agents], dim=0
        )

        _, _, _, P = batch

        for agent in self.hparams.agents:
            loss += agent.compute_loss(output[agent.idx])
            R[
                agent.idx * self.hparams.n : (agent.idx + 1) * self.hparams.n,
                agent.idx * self.hparams.n : (agent.idx + 1) * self.hparams.n,
            ] = agent.R

        REP = R @ E @ P

        for i in range(self.hparams.n):
            loss += (
                self.hparams.lmb
                * torch.linalg.norm(
                    self.hparams.B.T @ REP[i :: self.hparams.n, :]
                )
                ** 2
            )

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)

        return output, loss

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
        """Define the optimizer: Stochastic Gradient Descent.

        Returns:
            dict[str, object]
                The optimizer and scheduler.
        """
        optimizer = torch.optimizer.AdamW(
            self.parameters(), lr=self.hparams.LR
        )
        return {
            'optimizer': optimizer,
        }
