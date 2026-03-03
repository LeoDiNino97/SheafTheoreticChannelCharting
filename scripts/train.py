# train.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from src import CSIDataModule


@hydra.main(
    config_path="../config",
    config_name="train",
    version_base="1.3",
)
def main(cfg: DictConfig):

    # -----------------------------
    # Load DeepMIMO
    # -----------------------------
    dm = CSIDataModule(cfg.dataset)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    xA, xP, xN, y = batch
    print(xA.shape, xP.shape, None if xN is None else xN.shape, y)
    # -----------------------------
    # Model
    # -----------------------------
    # model = DummyCSIModel()

    # -----------------------------
    # Trainer
    # -----------------------------
    # trainer = L.Trainer(
    #     max_epochs=1,
    #     accelerator="auto",
    #     devices="auto",
    #     log_every_n_steps=1,
    # )

    # trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
