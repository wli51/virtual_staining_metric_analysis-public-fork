"""Deterministic reference-nucleus sampling, shared by all profiling backends."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from utils.fov_ingestion.writer import digest


@dataclass(frozen=True)
class SamplingPolicy:
    max_per_stratum: int
    strata: tuple[str, ...] = ('cell_type', 'seeding_density', 'plate')
    seed: int = 42
    allocation: str = 'fov_balanced'

    def descriptor(self):
        if not isinstance(self.max_per_stratum, int) or self.max_per_stratum < 1 or not self.strata or len(set(self.strata)) != len(self.strata):
            raise ValueError('Sampling requires a positive cap and distinct metadata columns')
        if self.allocation not in ('fov_balanced', 'uniform_instances'):
            raise ValueError('Unknown sampling allocation')
        return {**asdict(self), 'strata': list(self.strata), 'version': 'reference-sampling-v1'}


def sample_anchors(anchors, metadata, policy):
    """Cap each observed stratum; smaller strata retain all eligible nuclei.

    FOV-balanced allocation takes one randomly ranked nucleus per FOV per round,
    exhausting small FOVs before redistributing their unfilled quota. This is
    intentionally NOT a simple random sample of the pooled cell population.
    """
    policy.descriptor()
    missing = set(policy.strata) - set(metadata.columns)
    if missing:
        raise ValueError(f'Missing sampling metadata columns: {sorted(missing)}')
    meta = metadata[['fov_id', *policy.strata]].copy()
    if meta.fov_id.duplicated().any():
        raise ValueError('Duplicate FOV metadata')
    result = anchors.merge(meta, on='fov_id', how='left', validate='many_to_one')
    if result[list(policy.strata)].isna().any().any():
        raise ValueError('Missing stratum values; correct metadata before sampling')
    if result.object_id.duplicated().any():
        raise ValueError('Reference object IDs must be globally unique')
    result['crop_eligible'] = result.eligible.astype(bool)
    result['selected'] = False
    result['sampling_rank'] = pd.Series(pd.NA, index=result.index, dtype='Int64')
    result['stratum_id'] = [digest([str(v) for v in row]) for row in result[list(policy.strata)].itertuples(index=False, name=None)]
    for _, group in result.loc[result.crop_eligible].groupby('stratum_id', sort=True):
        group = group.copy()
        group['_object_rank'] = [digest([policy.seed, oid]) for oid in group.object_id]
        if policy.allocation == 'fov_balanced':
            group['_fov_rank'] = [digest([policy.seed, fid]) for fid in group.fov_id]
            group = group.sort_values(['_object_rank', 'object_id'])
            group['_round'] = group.groupby('fov_id').cumcount()
            group = group.sort_values(['_round', '_fov_rank', 'fov_id'])
        else:
            group = group.sort_values(['_object_rank', 'object_id'])
        result.loc[group.index, 'sampling_rank'] = np.arange(1, len(group) + 1)
        result.loc[group.index[:policy.max_per_stratum], 'selected'] = True
    result.loc[result.crop_eligible & ~result.selected, 'exclusion_reason'] = 'stratified_subsample'
    result['eligible'] = result.selected
    return result.sort_values(['fov_id', 'object_id']).reset_index(drop=True)


def reuse_plan(anchors, path):
    """Reuse exact object selection; reject changed cohort, masks or crop policy."""
    saved = pd.read_parquet(path).sort_values(['fov_id', 'object_id']).reset_index(drop=True)
    current = anchors.sort_values(['fov_id', 'object_id']).reset_index(drop=True)
    required = {'selected', 'crop_eligible'}
    if not required.issubset(saved.columns):
        raise ValueError('Not a reference sampling plan')
    restored = saved[current.columns].copy()
    restored['eligible'] = saved.crop_eligible
    restored.loc[saved.crop_eligible, 'exclusion_reason'] = ''
    pd.testing.assert_frame_equal(restored, current, check_dtype=False)
    if saved.selected.isna().any() or not saved.selected.isin([True, False]).all():
        raise ValueError('Invalid selected flags')
    if (saved.selected & ~saved.crop_eligible).any() or not saved.eligible.equals(saved.selected):
        raise ValueError('Sampling plan includes ineligible anchors')
    return saved


def sampling_summary(plan):
    from .crops import ANCHOR_COLUMNS
    excluded = set(ANCHOR_COLUMNS) | {'crop_eligible', 'selected', 'sampling_rank', 'reference_fingerprint', 'stratum_id'}
    metadata_columns = [c for c in plan if c not in excluded]
    summary = plan.groupby('stratum_id', sort=True).agg(
        n_candidates=('object_id', 'size'), n_crop_eligible=('crop_eligible', 'sum'),
        n_selected=('selected', 'sum'), n_fovs=('fov_id', 'nunique'),
    ).reset_index()
    return summary.merge(plan[['stratum_id', *metadata_columns]].drop_duplicates('stratum_id'), on='stratum_id', validate='one_to_one')
