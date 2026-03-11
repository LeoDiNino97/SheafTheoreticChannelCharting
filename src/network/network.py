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

    def training_step(self, batch, batch_idx):
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

        # TODO Train-loss logging
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def _shared_eval(self, batch, batch_idx, prefix):
        pass

    def configure_optimizers(self):
        return torch.optimizer.AdamW(self.parameters(), lr=self.hparams.LR)
