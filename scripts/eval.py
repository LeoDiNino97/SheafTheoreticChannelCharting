"""Evaluation script: load saved checkpoints and compute metrics for all orchestrators.

Uses a dedicated ``eval.yaml`` Hydra config.  For each orchestrator type the
script:

1. Finds the checkpoint with the lowest encoded training loss.
2. Loads the full orchestrator once.
3. Runs ``num_folds`` evaluation passes, each with a different ``triplet_seed``
   drawn from ``cfg.triplet_seeds``.  Each pass produces one independent set
   of topology metrics — effectively a bootstrap over data resampling.
4. Computes mean and std of KS, CT@K, and TW@K across folds, both globally
   (averaged over agents) and per agent.

Per-orchestrator results are written to:
    ``results/{subdir}/{orch_name}/eval_metrics.parquet``

An aggregated summary across all orchestrators is also written to:
    ``results/{subdir}/eval_metrics.parquet``
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import polars as pl
import torch
from lightning import seed_everything
from omegaconf import DictConfig

from scripts.util import (
    compute_eval_metrics,
    find_best_checkpoint,
    get_checkpoint_dir,
    get_results_dir,
    save_orch_metrics,
)
from src.datamodules.dichasus import DICHASUSDataModule

ORCHESTRATOR_NAMES = [
    'bundle',
    'cover_sheaf',
    'diag_sheaf',
    'federated',
    'flat_bundle',
    'optimal_transport',
    'personalized_federated',
    'vanilla',
]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float('nan')


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return float('nan')
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


@hydra.main(
    config_path='../config/hydra/',
    config_name='eval',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    CURRENT = Path('.')

    ckpt_dir = get_checkpoint_dir(cfg, CURRENT)
    results_subdir = get_results_dir(cfg, CURRENT)
    results_subdir.mkdir(exist_ok=True, parents=True)

    project_name = cfg.logger.project

    K_max: int = cfg.get('eval_K_max', 40)
    K_min: int = cfg.get('eval_K_min', 2)
    step: int = cfg.get('eval_K_step', 4)

    num_folds: int = cfg.get('num_folds', 1)
    anchor_seed: int = cfg.get('anchor_seed', cfg.seed)
    triplet_seeds: list[int] = list(cfg.get('triplet_seeds', list(range(num_folds))))

    if num_folds > 1:
        results_subdir = results_subdir / 'multifold'
        results_subdir.mkdir(exist_ok=True, parents=True)

    records: list[dict] = []

    for orch_name in ORCHESTRATOR_NAMES:
        orch_dir = results_subdir / orch_name

        # Locate checkpoint
        ckpt_path = find_best_checkpoint(ckpt_dir, project_name, orch_name)
        if ckpt_path is None:
            print(f'[{orch_name}] No checkpoint found in {ckpt_dir} — skipping.')
            continue

        print(f'\n[{orch_name}] Loading {ckpt_path.name}')
        orchestrator = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        orchestrator.eval()

        # -------------------------------------------------------
        # Bootstrap evaluation: one pass per fold / triplet seed
        # -------------------------------------------------------
        # fold_metrics[fold_idx] = {'KS': [per-agent], 'CT': {K: [per-agent]}, 'TW': ...}
        fold_metrics: list[dict] = []

        for fold_idx, triplet_seed in enumerate(triplet_seeds):
            print(f'[{orch_name}] Fold {fold_idx + 1}/{num_folds}  (triplet_seed={triplet_seed})')

            datamodule = DICHASUSDataModule(
                cfg.dataset,
                anchor_seed=anchor_seed,
                triplet_seed=triplet_seed,
            )
            datamodule.prepare_data()
            datamodule.setup('fit')

            try:
                metrics = compute_eval_metrics(
                    orchestrator,
                    datamodule,
                    K_max=K_max,
                    K_min=K_min,
                    step=step,
                )
            except Exception as exc:
                print(f'[{orch_name}] Fold {fold_idx + 1} failed: {exc} — skipping fold.')
                continue

            fold_metrics.append(metrics)

        if not fold_metrics:
            print(f'[{orch_name}] All folds failed — skipping orchestrator.')
            continue

        n_folds = len(fold_metrics)

        # -------------------------------------------------------
        # Aggregate across folds: mean and std
        # -------------------------------------------------------
        row: dict = {'orchestrator': orch_name}

        # KS: per-agent mean/std across folds
        n_agents = len(fold_metrics[0]['KS'])
        for i in range(n_agents):
            vals = [fold_metrics[f]['KS'][i] for f in range(n_folds)]
            row[f'KS_agent_{i}'] = _mean(vals)
            row[f'KS_agent_{i}_std'] = _std(vals)

        # KS global mean/std (average agents first, then fold stats)
        fold_ks_means = [_mean(fold_metrics[f]['KS']) for f in range(n_folds)]
        row['KS_mean'] = _mean(fold_ks_means)
        row['KS_std'] = _std(fold_ks_means)

        # CT and TW: per-agent and global mean/std across folds
        for K in range(K_min, K_max + 1, step):
            for i in range(n_agents):
                ct_vals = [fold_metrics[f]['CT'][K][i] for f in range(n_folds)]
                tw_vals = [fold_metrics[f]['TW'][K][i] for f in range(n_folds)]
                row[f'CT_K{K}_agent_{i}'] = _mean(ct_vals)
                row[f'CT_K{K}_agent_{i}_std'] = _std(ct_vals)
                row[f'TW_K{K}_agent_{i}'] = _mean(tw_vals)
                row[f'TW_K{K}_agent_{i}_std'] = _std(tw_vals)

            # Global (agents averaged per fold, then fold stats)
            fold_ct_means = [_mean(fold_metrics[f]['CT'][K]) for f in range(n_folds)]
            fold_tw_means = [_mean(fold_metrics[f]['TW'][K]) for f in range(n_folds)]
            row[f'CT_K{K}'] = _mean(fold_ct_means)
            row[f'CT_K{K}_std'] = _std(fold_ct_means)
            row[f'TW_K{K}'] = _mean(fold_tw_means)
            row[f'TW_K{K}_std'] = _std(fold_tw_means)

        # FOSCTTM: mean/std across folds (global metric, not per-agent)
        foscttm_vals = [
            fold_metrics[f]['FOSCTTM']
            for f in range(n_folds)
            if fold_metrics[f].get('FOSCTTM') is not None
        ]
        row['FOSCTTM'] = _mean(foscttm_vals) if foscttm_vals else None
        row['FOSCTTM_std'] = _std(foscttm_vals) if len(foscttm_vals) > 1 else float('nan')

        print(
            f'[{orch_name}] KS={row["KS_mean"]:.4f}±{row["KS_std"]:.4f}  '
            f'CT_K10={row["CT_K10"]:.4f}±{row["CT_K10_std"]:.4f}  '
            f'FOSCTTM={row["FOSCTTM"]}'
        )

        save_orch_metrics(row, orch_dir)
        records.append(row)

    if not records:
        print('No orchestrators evaluated. Nothing to save.')
        return

    df = pl.DataFrame(records)
    agg_path = results_subdir / 'eval_metrics.parquet'
    df.write_parquet(agg_path)
    print(f'\nSaved aggregated metrics → {agg_path}')
    print(df)


if __name__ == '__main__':
    main()
