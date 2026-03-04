# datamodule.py
from typing import Any

import deepmimo as dm
import lightning as L
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .dataset import TrajectoryCSIDataset


def _merge_defaults(defaults: dict[str, Any], override: Any) -> dict[str, Any]:
    if isinstance(override, DictConfig):
        override = OmegaConf.to_container(override, resolve=True)
    override = override or {}
    out = dict(defaults)
    out.update(override)
    return out


class CSIDataModule(L.LightningDataModule):
    DEFAULTS: dict[str, Any] = {
        'scenario': 'asu_campus_3p5',
        'download': True,
        'batch_size': 64,
        'num_workers': 0,
        'shuffle': True,
        'pin_memory': True,
        'compute_channels': {},
        'num_users': 200,
        'T_min': 20,
        'T_max': 60,
        'pair_mode': 'triplet',  # "triplet" or "contrastive"
        'window': 3,
        'include_same_user_outside_window': False,
        'p_positive': 0.5,  # for contrastive
        'train_seed': 27,
        'test_seed': 42,
        'val_seed': 123,
        'train_split': 0.8,
        'val_split': 0.2,
    }

    def __init__(self, dataset_cfg: DictConfig | dict[str, Any]):
        super().__init__()
        self.cfg = _merge_defaults(self.DEFAULTS, dataset_cfg)
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        assert 0 <= self.cfg['train_split'] <= 1, (
            'The "train_split" must be between 0 and 1.'
        )
        assert 0 <= self.cfg['val_split'] <= 1, (
            'The "val_split" must be between 0 and 1.'
        )

    def prepare_data(self) -> None:
        if self.cfg['download'] and hasattr(dm, 'download'):
            dm.download(self.cfg['scenario'])

    def setup(self, stage: str | None = None) -> None:
        ds = dm.load(self.cfg['scenario'])
        ds0 = ds[0] if isinstance(ds, (list, tuple)) else ds

        ch_kwargs = self.cfg.get('compute_channels') or {}
        try:
            ds0.compute_channels(**ch_kwargs)
        except TypeError:
            ds0.compute_channels()

        ch = ds0.channels
        if ch is None:
            raise RuntimeError(
                'DeepMIMO: ds0.channels is None after compute_channels().'
            )

        per_sample_complex = int(np.prod(ch.shape[1:]))
        self.feature_dim = 2 * per_sample_complex

        train_num_users = int(
            int(self.cfg['num_users']) * (self.cfg['train_split'])
        )
        test_num_users = int(
            int(self.cfg['num_users']) * (1 - self.cfg['train_split'])
        )
        val_num_users = int(
            int(self.cfg['num_users']) * (self.cfg['val_split'])
        )

        self.train_dataset = TrajectoryCSIDataset(
            rx_pos=ds0.rx_pos,
            H_users=ds0.channels,
            num_users=train_num_users,
            T_min=int(self.cfg['T_min']),
            T_max=int(self.cfg['T_max']),
            seed=int(self.cfg['train_seed']),
            pair_mode=str(self.cfg['pair_mode']),
            window=int(self.cfg['window']),
            include_same_user_outside_window=bool(
                self.cfg['include_same_user_outside_window']
            ),
            p_positive=float(self.cfg['p_positive']),
        )

        self.test_dataset = TrajectoryCSIDataset(
            rx_pos=ds0.rx_pos,
            H_users=ds0.channels,
            num_users=test_num_users,
            T_min=int(self.cfg['T_min']),
            T_max=int(self.cfg['T_max']),
            seed=int(self.cfg['test_seed']),
            pair_mode=str(self.cfg['pair_mode']),
            window=int(self.cfg['window']),
            include_same_user_outside_window=bool(
                self.cfg['include_same_user_outside_window']
            ),
            p_positive=float(self.cfg['p_positive']),
        )

        self.val_dataset = TrajectoryCSIDataset(
            rx_pos=ds0.rx_pos,
            H_users=ds0.channels,
            num_users=val_num_users,
            T_min=int(self.cfg['T_min']),
            T_max=int(self.cfg['T_max']),
            seed=int(self.cfg['val_seed']),
            pair_mode=str(self.cfg['pair_mode']),
            window=int(self.cfg['window']),
            include_same_user_outside_window=bool(
                self.cfg['include_same_user_outside_window']
            ),
            p_positive=float(self.cfg['p_positive']),
        )

        return None

    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            DataLoader
                The train DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg['batch_size']),
            shuffle=bool(self.cfg['shuffle']),
            num_workers=int(self.cfg['num_workers']),
            pin_memory=bool(self.cfg['pin_memory']),
        )

    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            DataLoader
                The test DataLoader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.cfg['batch_size']),
            shuffle=False,
            num_workers=int(self.cfg['num_workers']),
            pin_memory=bool(self.cfg['pin_memory']),
        )

    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            DataLoader
                The validation DataLoader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg['batch_size']),
            shuffle=False,
            num_workers=int(self.cfg['num_workers']),
            pin_memory=bool(self.cfg['pin_memory']),
        )
