"""
Lightning DataModule for generating trajectory-based
CSI datasets using DeepMIMO.

This module:
1. Downloads/loads a DeepMIMO scenario.
2. Computes channel matrices.
3. Creates synthetic user trajectories from the available receiver positions.
4. Builds train/validation/test datasets using TrajectoryCSIDataset.
5. Provides PyTorch DataLoaders for training pipelines.

The DataModule follows the PyTorch Lightning lifecycle:
    prepare_data() -> setup() -> train/val/test_dataloader()
"""

from typing import Any

import deepmimo as dm
import lightning as L
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .dataset import TrajectoryCSIDataset


def _merge_defaults(
    defaults: dict[str, Any],
    override: Any,
) -> dict[str, Any]:
    """
    Merge default configuration values with user-provided overrides.

    This utility ensures that:
    - If the user passes an OmegaConf DictConfig,
      it is converted to a standard dict.
    - Missing parameters fall back to the provided defaults.

    Parameters
    ----------
    defaults : dict[str, Any]
        Default configuration dictionary.
    override : Any
        User-provided configuration. Can be:
        - dict
        - OmegaConf DictConfig
        - None

    Returns
    -------
    dict[str, Any]
        A merged configuration dictionary where `override`
        values take precedence over defaults.
    """
    # Convert OmegaConf configuration into a standard Python dictionary
    if isinstance(override, DictConfig):
        override = OmegaConf.to_container(override, resolve=True)

    # If override is None, treat it as an empty dictionary
    override = override or {}

    # Create a copy of defaults and update with overrides
    out = dict(defaults)
    out.update(override)

    return out


class CSIDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for trajectory-based CSI learning.

    This module wraps the DeepMIMO dataset and converts it into
    trajectory-based samples suitable for contrastive or triplet
    learning tasks on channel state information (CSI).

    Responsibilities:
    ----------------
    - Download/load a DeepMIMO scenario
    - Compute channel matrices
    - Generate trajectory datasets
    - Split datasets into train / validation / test
    - Provide DataLoaders for training pipelines

    The resulting datasets produce CSI samples arranged as:
        real/imaginary concatenated channel vectors.

    Attributes
    ----------
    cfg : dict
        Final merged configuration.
    train_dataset : TrajectoryCSIDataset | None
        Training dataset instance.
    test_dataset : TrajectoryCSIDataset | None
        Test dataset instance.
    val_dataset : TrajectoryCSIDataset | None
        Validation dataset instance.
    feature_dim : int
        Dimension of flattened CSI feature vectors (real + imaginary parts).
    """

    # Default configuration values used if the user does not specify them
    DEFAULTS: dict[str, Any] = {
        'scenario': 'asu_campus_3p5',  # DeepMIMO scenario name
        'download': True,  # whether to download the dataset if missing
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

    def __init__(
        self,
        dataset_cfg: DictConfig | dict[str, Any],
    ):
        """
        Initialize the CSIDataModule.

        Parameters
        ----------
        dataset_cfg : DictConfig | dict[str, Any]
            User configuration overriding DEFAULTS.
            Can be provided via Hydra/OmegaConf or as a normal dictionary.
        """
        super().__init__()

        # Merge user configuration with defaults
        self.cfg = _merge_defaults(self.DEFAULTS, dataset_cfg)

        # Dataset placeholders (initialized during setup)
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        # Basic validation of split parameters
        assert 0 <= self.cfg['train_split'] <= 1, (
            'The "train_split" must be between 0 and 1.'
        )
        assert 0 <= self.cfg['val_split'] <= 1, (
            'The "val_split" must be between 0 and 1.'
        )

    def prepare_data(self) -> None:
        """
        Download the DeepMIMO scenario if required.

        This method is executed **once per node** in distributed setups
        and should only contain operations that are safe to run once,
        such as dataset downloads.

        Returns
        -------
        None
        """
        # Only download if explicitly requested and supported
        if self.cfg['download'] and hasattr(dm, 'download'):
            dm.download(self.cfg['scenario'])

        return None

    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        """
        Load the DeepMIMO dataset and construct trajectory datasets.

        This method:
        1. Loads the DeepMIMO scenario
        2. Computes channel matrices
        3. Determines CSI feature dimensionality
        4. Splits the dataset into train/test/validation sets
        5. Creates TrajectoryCSIDataset instances

        Parameters
        ----------
        stage : str | None
            Stage of the Lightning lifecycle ("fit", "test", etc.).
            Currently not used but kept for Lightning compatibility.

        Returns
        -------
        None
        """

        # Load DeepMIMO scenario
        ds = dm.load(self.cfg['scenario'])

        # Some scenarios return a list of datasets
        # (e.g., multiple base stations)
        ds0 = ds[0] if isinstance(ds, (list, tuple)) else ds

        # Channel computation arguments
        ch_kwargs = self.cfg.get('compute_channels') or {}

        # Compute channel matrices
        try:
            ds0.compute_channels(**ch_kwargs)
        except TypeError:
            # Fallback if the scenario does not accept arguments
            ds0.compute_channels()

        ch = ds0.channels

        # Ensure channel computation succeeded
        assert ch is not None, (
            'DeepMIMO: ds0.channels is None after compute_channels().'
        )

        # Compute feature dimensionality
        # Each channel is complex -> we convert to real/imag pairs
        per_sample_complex = int(np.prod(ch.shape[1:]))
        self.feature_dim = 2 * per_sample_complex

        # ---------------------------------------------------------------
        #                Compute dataset split sizes
        # ---------------------------------------------------------------

        train_num_users = int(
            int(self.cfg['num_users']) * (self.cfg['train_split'])
        )
        test_num_users = int(
            int(self.cfg['num_users']) * (1 - self.cfg['train_split'])
        )
        val_num_users = int(
            int(self.cfg['num_users']) * (self.cfg['val_split'])
        )

        # ---------------------------------------------------------------
        #                   Create the Datasets
        # ---------------------------------------------------------------

        # Training dataset
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

        # Test dataset
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

        # Validation dataset
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
        """
        Create the training DataLoader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader used during model training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg['batch_size']),
            shuffle=bool(self.cfg['shuffle']),
            num_workers=int(self.cfg['num_workers']),
            pin_memory=bool(self.cfg['pin_memory']),
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create the test DataLoader.

        Returns
        -------
        DataLoader
            DataLoader used during testing/evaluation.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.cfg['batch_size']),
            shuffle=False,
            num_workers=int(self.cfg['num_workers']),
            pin_memory=bool(self.cfg['pin_memory']),
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation DataLoader.

        Returns
        -------
        DataLoader
            DataLoader used during validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg['batch_size']),
            shuffle=False,
            num_workers=int(self.cfg['num_workers']),
            pin_memory=bool(self.cfg['pin_memory']),
        )
