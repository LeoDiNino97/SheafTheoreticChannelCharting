import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from torch.utils.data import Dataset


def csi_to_realvec(
    H: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a complex CSI tensor into a real-valued vector.

    Parameters
    ----------
    H : torch.Tensor
        Complex-valued CSI tensor, shape (...).

    Returns
    -------
    torch.Tensor
        Real-valued vector, shape (D,), where D = 2 * product of H dims
        except the first (sample) dimension.
    """
    # Flatten complex tensor into real vector (real + imag)
    return torch.view_as_real(H).reshape(-1).float()


class TrajectoryCSIDataset(Dataset):
    """
    Dataset producing Siamese batches for CSI trajectory learning.

    Supports two sampling modes:

    - 'triplet': returns (xA, xP, xN, y=-1)
    - 'contrastive': returns (xA, xP, xN=None, y in {0,1})

    Positives:
        Same user, |dt| <= window, excluding anchor.
    Negatives:
        Different user always, optionally same user outside window if
        include_same_user_outside_window=True.

    Parameters
    ----------
    rx_pos : np.ndarray
        Positions of receiver antennas, shape (N_rx, 2 or 3).
    H_users : np.ndarray
        CSI data per RX position, shape (N_rx, ...).
    num_users : int, optional
        Number of simulated users (default: 500).
    T_min : int, optional
        Minimum trajectory length (default: 32).
    T_max : int, optional
        Maximum trajectory length (default: 128).
    kinds : tuple, optional
        Trajectory types: 'linear', 'circular', 'random' (default all).
    pair_mode : str, optional
        'triplet' or 'contrastive' (default: 'triplet').
    window : int, optional
        Time window for positive sampling (default: 3).
    include_same_user_outside_window : bool, optional
        Include same-user points outside window as negatives
        (default: False).
    p_positive : float, optional
        Probability of positive pair in contrastive mode (default: 0.5).
    seed : int, optional
        Random seed (default: 0).

    Notes
    -----
    - Converts complex CSI to real-valued vectors with `csi_to_realvec`.
    - Builds variable-length trajectories per user at initialization.
    - Provides methods to sample positives and negatives efficiently.
    """

    def __init__(
        self,
        rx_pos: np.ndarray,
        H_users: np.ndarray,
        num_users: int = 500,
        T_min: int = 32,
        T_max: int = 128,
        kinds=('linear', 'circular', 'random'),
        linear_len=(20.0, 120.0),
        circle_r=(10.0, 60.0),
        random_step=(1.0, 5.0),
        random_keep_dir=0.7,
        z_min=None,
        z_max=None,
        seed: int = 0,
        # --- Siamese sampling controls ---
        pair_mode: str = 'triplet',  # "triplet" or "contrastive"
        window: int = 3,
        include_same_user_outside_window: bool = False,  # for negatives
        p_positive: float = 0.5,  # only for contrastive
    ):
        super().__init__()

        # Siamese sampling configuration
        assert pair_mode in ('triplet', 'contrastive')
        self.pair_mode = pair_mode
        self.window = int(window)
        self.include_same_user_outside_window = bool(
            include_same_user_outside_window
        )
        self.p_positive = float(p_positive)

        self.rng = np.random.default_rng(seed)

        # Convert RX positions to array, ensure 3D coordinates
        rx_pos = np.asarray(rx_pos, dtype=np.float64)
        if rx_pos.shape[1] == 2:
            rx_pos = np.c_[rx_pos, np.zeros((rx_pos.shape[0], 1))]
        self.rx_pos_all = rx_pos

        # Convert CSI to array and validate dimensions
        H_users = np.asarray(H_users)
        if H_users.shape[0] != rx_pos.shape[0]:
            raise ValueError('H_users first dim must match rx_pos first dim')
        self.H_users = H_users

        # Filter candidate RX points (optional)
        mask = np.ones(len(rx_pos), dtype=bool)
        if z_min is not None:
            mask &= rx_pos[:, 2] >= float(z_min)
        if z_max is not None:
            mask &= rx_pos[:, 2] <= float(z_max)

        self.valid_idxs = np.where(mask)[0]
        if len(self.valid_idxs) < 10:
            raise ValueError('Too few valid RX points after filtering.')
        self.rx_pos = rx_pos[self.valid_idxs]
        self.rx_xy = self.rx_pos[:, :2]
        self.kdtree = KDTree(self.rx_xy)

        self.num_users = int(num_users)
        self.T_min = int(T_min)
        self.T_max = int(T_max)
        if self.T_min < 2 or self.T_max < self.T_min:
            raise ValueError('Bad T_min/T_max')
        self.kinds = tuple(kinds)

        self.linear_len = linear_len
        self.circle_r = circle_r
        self.random_step = random_step
        self.random_keep_dir = float(random_keep_dir)

        # ---- Build variable-length trajectories once ----
        self._user_of_index = []
        self._t_of_index = []
        self._rx_of_index = []

        for user_id in range(self.num_users):
            # Random trajectory length
            T = int(self.rng.integers(self.T_min, self.T_max + 1))

            # Randomly pick trajectory kind
            kind = self.kinds[int(self.rng.integers(0, len(self.kinds)))]

            # Generate trajectory of RX indices
            rx_idxs = self._generate_one(kind, T)

            # Store per-step info for sampling
            for t, rx in enumerate(rx_idxs):
                self._user_of_index.append(user_id)
                self._t_of_index.append(t)
                self._rx_of_index.append(int(rx))

        # Convert to arrays for fast indexing
        self._user_of_index = np.asarray(self._user_of_index, dtype=np.int64)
        self._t_of_index = np.asarray(self._t_of_index, dtype=np.int64)
        self._rx_of_index = np.asarray(self._rx_of_index, dtype=np.int64)

        # Map user -> global indices (contiguous, but keep general)
        self.user_to_indices = {}
        for uid in range(self.num_users):
            self.user_to_indices[uid] = np.where(self._user_of_index == uid)[
                0
            ].astype(np.int64)

    # ----------------- Snapping helpers -----------------
    def _snap(
        self,
        xy: np.ndarray,
    ) -> np.ndarray:
        """
        Snap 2D coordinates to the nearest valid RX index.

        Parameters
        ----------
        xy : np.ndarray
            XY coordinates of shape (2,) or (N,2).

        Returns
        -------
        np.ndarray
            Indices of nearest valid RX positions.
        """
        _, idx_local = self.kdtree.query(xy, k=1)
        return self.valid_idxs[idx_local].astype(np.int64)

    # ----------------- Trajectory generators -----------------
    def _rand_anchor_xy(self) -> np.ndarray:
        """
        Pick a random RX XY coordinate to serve as a trajectory anchor.

        Returns
        -------
        np.ndarray
            Selected XY coordinate (2,).
        """
        return self.rx_xy[int(self.rng.integers(0, len(self.rx_xy)))].copy()

    def _generate_one(
        self,
        kind: str,
        T: int,
    ) -> np.ndarray:
        """
        Generate a trajectory of length T of the specified kind.

        Parameters
        ----------
        kind : str
            One of 'linear', 'circular', or 'random'.
        T : int
            Trajectory length.

        Returns
        -------
        np.ndarray
            Array of RX indices representing the trajectory.
        """
        if kind == 'linear':
            start = self._rand_anchor_xy()
            L = float(self.rng.uniform(*self.linear_len))
            ang = float(self.rng.uniform(0, 2 * np.pi))
            end = start + L * np.array([np.cos(ang), np.sin(ang)])
            s = np.linspace(0.0, 1.0, T)
            xy = start * (1 - s)[:, None] + end * s[:, None]
            return self._snap(xy)

        if kind == 'circular':
            center = self._rand_anchor_xy()
            r = float(self.rng.uniform(*self.circle_r))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            ang = np.linspace(0.0, 2 * np.pi, T, endpoint=False) + phase
            xy = np.stack(
                [center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)],
                axis=1,
            )
            return self._snap(xy)

        if kind == 'random':
            xy = np.empty((T, 2), dtype=np.float64)
            xy[0] = self._rand_anchor_xy()
            prev_dir = None
            for t in range(1, T):
                step = float(self.rng.uniform(*self.random_step))
                if (
                    prev_dir is None
                    or self.rng.random() > self.random_keep_dir
                ):
                    ang = float(self.rng.uniform(0, 2 * np.pi))
                else:
                    ang = float(
                        np.arctan2(prev_dir[1], prev_dir[0])
                        + self.rng.normal(0, 0.5)
                    )
                d = np.array([np.cos(ang), np.sin(ang)])
                xy[t] = xy[t - 1] + step * d
                prev_dir = d
            return self._snap(xy)

        raise ValueError(f'Unknown kind: {kind}')

    # ----------------- mining -----------------
    def get_positive_examples(
        self,
        anchor_idx: int,
        window: int,
    ) -> np.ndarray:
        """
        Return indices of positive samples for the given anchor.

        Positives are from the same user and within +/- window steps
        around the anchor, excluding the anchor itself.

        Parameters
        ----------
        anchor_idx : int
            Index of the anchor sample.
        window : int
            Time window for selecting positives.

        Returns
        -------
        np.ndarray
            Indices of positive samples.
        """
        uid = int(self._user_of_index[anchor_idx])
        t0 = int(self._t_of_index[anchor_idx])
        user_indices = self.user_to_indices[uid]
        user_ts = self._t_of_index[user_indices]
        m = (user_ts >= t0 - window) & (user_ts <= t0 + window)
        pos = user_indices[m]
        return pos[pos != anchor_idx]

    def get_negative_examples(
        self,
        anchor_idx: int,
        window: int,
        include_same_user_outside_window: bool = True,
    ) -> np.ndarray:
        """
        Return indices of negative samples for the given anchor.

        Negatives include all samples from different users. Optionally,
        same-user samples outside the window can also be included.

        Parameters
        ----------
        anchor_idx : int
            Index of the anchor sample.
        window : int
            Time window for positive samples (used to exclude near positives).
        include_same_user_outside_window : bool
            Whether to include same-user samples outside the window.

        Returns
        -------
        np.ndarray
            Indices of negative samples.
        """
        uid = int(self._user_of_index[anchor_idx])
        t0 = int(self._t_of_index[anchor_idx])

        # different user always negative
        neg = np.where(self._user_of_index != uid)[0].astype(np.int64)

        if include_same_user_outside_window:
            user_indices = self.user_to_indices[uid]
            user_ts = self._t_of_index[user_indices]
            m_out = (user_ts < t0 - window) | (user_ts > t0 + window)
            neg = np.concatenate([neg, user_indices[m_out]], axis=0)

        return neg

    # ----------------- CSI helpers -----------------
    def _H_from_global_index(self, gidx: int) -> torch.Tensor:
        """
        Return the CSI tensor for the given global index.

        Parameters
        ----------
        gidx : int
            Global index into the flattened trajectory dataset.

        Returns
        -------
        torch.Tensor
            Complex CSI tensor for the corresponding RX location.
        """
        rx_idx = int(self._rx_of_index[gidx])
        return torch.from_numpy(self.H_users[rx_idx])  # complex tensor

    def _pick_one(self, idxs: np.ndarray) -> int:
        """
        Randomly select one index from a list of indices.

        Parameters
        ----------
        idxs : np.ndarray
            Array of candidate indices.

        Returns
        -------
        int
            Randomly selected index, or -1 if input is empty.
        """
        if idxs is None or len(idxs) == 0:
            return -1
        return int(idxs[int(self.rng.integers(0, len(idxs)))])

    # ----------------- Dataset API -----------------
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of trajectory points across all users.
        """
        return int(self._user_of_index.shape[0])

    def __getitem__(
        self,
        index: int,
    ):
        """
        Return one Siamese sample depending on pair_mode.

        Returns
        -------
        xA, xP, xN, y
            CSI vectors and label / placeholder.
        """
        # Anchor
        H_A = self._H_from_global_index(index)

        pos = self.get_positive_examples(index, self.window)
        neg = self.get_negative_examples(
            index, self.window, self.include_same_user_outside_window
        )

        p_idx = self._pick_one(pos)
        if p_idx < 0:
            p_idx = index  # degenerate fallback

        # Triplet
        if self.pair_mode == 'triplet':
            n_idx = self._pick_one(neg)
            if n_idx < 0:
                n_idx = index

            H_P = self._H_from_global_index(p_idx)
            H_N = self._H_from_global_index(n_idx)

            xA = csi_to_realvec(H_A)
            xP = csi_to_realvec(H_P)
            xN = csi_to_realvec(H_N)

            y = torch.tensor(-1, dtype=torch.long)  # <-- IMPORTANT
            return xA, xP, xN, y

        # Contrastive:
        # sample positive pair with prob p_positive else negative pair
        if self.rng.random() < self.p_positive:
            H_P = self._H_from_global_index(p_idx)
            xA = csi_to_realvec(H_A)
            xP = csi_to_realvec(H_P)
            xN = None
            y = torch.tensor(1, dtype=torch.long)
        else:
            n_idx = self._pick_one(neg)
            if n_idx < 0:
                n_idx = index
            H_N = self._H_from_global_index(n_idx)
            xA = csi_to_realvec(H_A)
            xP = csi_to_realvec(H_N)  # second element of pair
            xN = None
            y = torch.tensor(0, dtype=torch.long)
        return xA, xP, xN, y
