"""Matched feature differences, streamed by FOV; no phenotype-loss claim implied."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.fov_ingestion.writer import digest


def iter_comparisons(run):
    run = Path(run)
    spec = json.loads((run / 'run.json').read_text())
    if (run / '_SUCCESS').read_text().strip() != digest(spec):
        raise ValueError('Profiling run is incomplete')
    if spec['profiler']['name'] == 'cp_measure':
        raise ValueError('CP profiles mix units and scopes; use iter_cp_profiles and compare named features or reference-scaled matrices')
    manifest = pd.read_parquet(run / 'manifest.parquet')
    reference_id = spec['sources'][0]['source_id']
    keys = ['object_id', 'compartment', 'crop_id', 'channel_key']
    for fid in spec['fov_ids']:
        rows = manifest.loc[manifest.fov_id.eq(fid)].set_index('source_id')
        reference = pd.read_parquet(run / rows.loc[reference_id, 'feature_uri'])
        reference['channel_key'] = reference.channels.map(lambda x: json.dumps(list(x)))
        for source in spec['sources'][1:]:
            derived = pd.read_parquet(run / rows.loc[source['source_id'], 'feature_uri'])
            derived['channel_key'] = derived.channels.map(lambda x: json.dumps(list(x)))
            paired = reference.merge(derived, on=keys, suffixes=('_reference', '_derived'),
                                     how='outer', validate='one_to_one', indicator=True)
            if not paired['_merge'].eq('both').all():
                raise ValueError('Reference/variant observation cohorts differ')
            if paired.empty:
                continue
            if not (paired.reference_image_ids_reference.map(list) == paired.reference_image_ids_derived.map(list)).all():
                raise ValueError('Reference image IDs differ in paired features')
            a, b = np.stack(paired.features_reference).astype(np.float64), np.stack(paired.features_derived).astype(np.float64)
            norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (norm_a > 0) & (norm_b > 0)
            cosine = np.full(len(a), np.nan)
            cosine[valid] = 1 - np.clip(np.sum(a[valid] * b[valid], axis=1) / (norm_a[valid] * norm_b[valid]), -1, 1)
            l2 = np.linalg.norm(b - a, axis=1)
            relative = np.divide(l2, norm_a, out=np.full(len(a), np.nan), where=norm_a > 0)
            yield pd.DataFrame({
                'profile_run_id': spec['profile_run_id'], 'feature_set_id': spec['feature_set_id'],
                'fov_id': fid, 'source_id': source['source_id'], 'variant_id': source['variant_id'],
                'object_id': paired.object_id, 'compartment': paired.compartment,
                'crop_id': paired.crop_id, 'channel_key': paired.channel_key,
                'cosine_distance': cosine, 'l2_distance': l2, 'relative_l2': relative,
                'reference_norm': norm_a, 'derived_norm': norm_b,
                'cosine_valid': valid,
            })


def write_comparisons(run, output):
    output = Path(output)
    if output.suffix != '.parquet':
        raise ValueError('Comparison output must be a .parquet file')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.tmp')
    schema = pa.schema([(k, pa.string()) for k in ['profile_run_id', 'feature_set_id', 'fov_id', 'source_id',
                       'variant_id', 'object_id', 'compartment', 'crop_id', 'channel_key']] +
                       [(k, pa.float64()) for k in ['cosine_distance', 'l2_distance', 'relative_l2',
                        'reference_norm', 'derived_norm']] + [('cosine_valid', pa.bool_())])
    n = 0
    try:
        with pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
            for frame in iter_comparisons(run):
                writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
                n += len(frame)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return n
