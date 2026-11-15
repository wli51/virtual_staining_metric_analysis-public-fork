"""Read named CP profiles without treating structural NaNs as measurements."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.fov_ingestion.writer import digest


def iter_cp_profiles(run, *, scope='single_channel', compartment=None, variant_id=None):
    """Yield one named-feature DataFrame per shard; keep original metadata columns.

    scope: geometry (channels=[]), single_channel, or channel_pair. Pair channel
    order follows run.json and the directional measurements' _1/_2 suffixes.
    """
    run = Path(run)
    spec = json.loads((run / 'run.json').read_text())
    if (run / '_SUCCESS').read_text().strip() != digest(spec):
        raise ValueError('Profiling run is incomplete')
    if spec['profiler']['name'] != 'cp_measure':
        raise ValueError('Expected a CP profiling run')
    arity = {'geometry': 0, 'single_channel': 1, 'channel_pair': 2}[scope]
    catalog = pd.DataFrame(spec['profiler']['feature_catalog'])
    catalog = catalog.loc[catalog.scope.eq(scope)].sort_values('feature_index')
    if catalog.empty:
        return
    manifest = pd.read_parquet(run / 'manifest.parquet')
    source_ids = {s['source_id'] for s in spec['sources'] if variant_id is None or s['variant_id'] == variant_id}
    for row in manifest.loc[manifest.source_id.isin(source_ids)].itertuples():
        frame = pd.read_parquet(run / row.feature_uri)
        frame = frame.loc[frame.channels.map(len).eq(arity)]
        if compartment is not None:
            frame = frame.loc[frame.compartment.eq(compartment)]
        if frame.empty:
            continue
        values = np.stack(frame.features)[:, catalog.feature_index.to_numpy()]
        metadata = frame.drop(columns='features').reset_index(drop=True)
        yield pd.concat([metadata, pd.DataFrame(values, columns=catalog.feature_name)], axis=1)
