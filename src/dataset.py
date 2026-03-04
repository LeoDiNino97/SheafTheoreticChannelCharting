import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from torch.utils.data import Dataset


def csi_to_realvec(H: torch.Tensor) -> torch.Tensor:
    """
    Convert complex CSI tensor (...) -> float vector (D,)
    """
    return torch.view_as_real(H).reshape(-1).float()


class TrajectoryCSIDataset(Dataset):
    """
    Produces Siamese batches directly.

    triplet:    returns (xA, xP, xN, y=None)
    contrastive returns (xA, xP, xN=None, y in {0,1})

    Positives: same user_id, |dt|<=window, excluding anchor
    Negatives: different user_id (always)
               optionally include same-user but |dt|>window if include_same_user_outside_window=True
    """

    def __init__(
        self,
        rx_pos: np.ndarray,
        H_users: np.ndarray,
        num_users: int = 500,
        T_min: int = 32,
        T_max: int = 128,
        kinds=("linear", "circular", "random"),
        linear_len=(20.0, 120.0),
        circle_r=(10.0, 60.0),
        random_step=(1.0, 5.0),
        random_keep_dir=0.7,
        z_min=None,
        z_max=None,
        seed: int = 0,
        # --- Siamese sampling controls ---
        pair_mode: str = "triplet",  # "triplet" or "contrastive"
        window: int = 3,
        include_same_user_outside_window: bool = False,  # for negatives
        p_positive: float = 0.5,  # only for contrastive
    ):
        super().__init__()
        assert pair_mode in ("triplet", "contrastive")
        self.pair_mode = pair_mode
        self.window = int(window)
        self.include_same_user_outside_window = bool(include_same_user_outside_window)
        self.p_positive = float(p_positive)

        self.rng = np.random.default_rng(seed)

        rx_pos = np.asarray(rx_pos, dtype=np.float64)
        if rx_pos.shape[1] == 2:
            rx_pos = np.c_[rx_pos, np.zeros((rx_pos.shape[0], 1))]
        self.rx_pos_all = rx_pos

        H_users = np.asarray(H_users)
        if H_users.shape[0] != rx_pos.shape[0]:
            raise ValueError("H_users first dim must match rx_pos first dim")
        self.H_users = H_users

        # Filter candidate RX points (optional)
        mask = np.ones(len(rx_pos), dtype=bool)
        if z_min is not None:
            mask &= rx_pos[:, 2] >= float(z_min)
        if z_max is not None:
            mask &= rx_pos[:, 2] <= float(z_max)

        self.valid_idxs = np.where(mask)[0]
        if len(self.valid_idxs) < 10:
            raise ValueError("Too few valid RX points after filtering.")
        self.rx_pos = rx_pos[self.valid_idxs]
        self.rx_xy = self.rx_pos[:, :2]
        self.kdtree = KDTree(self.rx_xy)

        self.num_users = int(num_users)
        self.T_min = int(T_min)
        self.T_max = int(T_max)
        if self.T_min < 2 or self.T_max < self.T_min:
            raise ValueError("Bad T_min/T_max")
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
            T = int(self.rng.integers(self.T_min, self.T_max + 1))
            kind = self.kinds[int(self.rng.integers(0, len(self.kinds)))]
            rx_idxs = self._generate_one(kind, T)

            for t, rx in enumerate(rx_idxs):
                self._user_of_index.append(user_id)
                self._t_of_index.append(t)
                self._rx_of_index.append(int(rx))

        self._user_of_index = np.asarray(self._user_of_index, dtype=np.int64)
        self._t_of_index = np.asarray(self._t_of_index, dtype=np.int64)
        self._rx_of_index = np.asarray(self._rx_of_index, dtype=np.int64)

        # user -> global indices (contiguous, but keep general)
        self.user_to_indices = {}
        for uid in range(self.num_users):
            self.user_to_indices[uid] = np.where(self._user_of_index == uid)[0].astype(
                np.int64
            )

    # ----------------- snapping -----------------
    def _snap(self, xy: np.ndarray) -> np.ndarray:
        _, idx_local = self.kdtree.query(xy, k=1)
        return self.valid_idxs[idx_local].astype(np.int64)

    # ----------------- generators -----------------
    def _rand_anchor_xy(self) -> np.ndarray:
        return self.rx_xy[int(self.rng.integers(0, len(self.rx_xy)))].copy()

    def _generate_one(self, kind: str, T: int) -> np.ndarray:
        if kind == "linear":
            start = self._rand_anchor_xy()
            L = float(self.rng.uniform(*self.linear_len))
            ang = float(self.rng.uniform(0, 2 * np.pi))
            end = start + L * np.array([np.cos(ang), np.sin(ang)])
            s = np.linspace(0.0, 1.0, T)
            xy = start * (1 - s)[:, None] + end * s[:, None]
            return self._snap(xy)

        if kind == "circular":
            center = self._rand_anchor_xy()
            r = float(self.rng.uniform(*self.circle_r))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            ang = np.linspace(0.0, 2 * np.pi, T, endpoint=False) + phase
            xy = np.stack(
                [center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)], axis=1
            )
            return self._snap(xy)

        if kind == "random":
            xy = np.empty((T, 2), dtype=np.float64)
            xy[0] = self._rand_anchor_xy()
            prev_dir = None
            for t in range(1, T):
                step = float(self.rng.uniform(*self.random_step))
                if prev_dir is None or self.rng.random() > self.random_keep_dir:
                    ang = float(self.rng.uniform(0, 2 * np.pi))
                else:
                    ang = float(
                        np.arctan2(prev_dir[1], prev_dir[0]) + self.rng.normal(0, 0.5)
                    )
                d = np.array([np.cos(ang), np.sin(ang)])
                xy[t] = xy[t - 1] + step * d
                prev_dir = d
            return self._snap(xy)

        raise ValueError(f"Unknown kind: {kind}")

    # ----------------- mining -----------------
    def get_positive_examples(self, anchor_idx: int, window: int) -> np.ndarray:
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

    def _H_from_global_index(self, gidx: int) -> torch.Tensor:
        rx_idx = int(self._rx_of_index[gidx])
        return torch.from_numpy(self.H_users[rx_idx])  # complex tensor

    def _pick_one(self, idxs: np.ndarray) -> int:
        if idxs is None or len(idxs) == 0:
            return -1
        return int(idxs[int(self.rng.integers(0, len(idxs)))])

    # ----------------- Dataset API -----------------
    def __len__(self) -> int:
        return int(self._user_of_index.shape[0])

    def __getitem__(self, index: int):
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
        if self.pair_mode == "triplet":
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

        # Contrastive: sample positive pair with prob p_positive else negative pair
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
