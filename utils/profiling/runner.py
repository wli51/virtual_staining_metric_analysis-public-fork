"""One reference crop cohort, multiple image collections, one reusable profiler."""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.writer import digest

from .api import FOVContext
from .crops import CropPolicy, make_anchors
from .sampling import reuse_plan, sample_anchors, sampling_summary
from .warning_log import logged_cp_batches
from .writer import atomic_json, feature_schema, feature_table, merge_parquet, run_lock


def _collection(value):
    return value if isinstance(value, MicroscopyCollection) else MicroscopyCollection(value)


def profile_collections(reference, variants, output, profiler, *, channels=None, fov_ids=None,
                        crop_policy=None, batch_size=128, dry_run=False,
                        sampling_policy=None, sampling_plan=None, workers=1, threads_per_worker=1):
    """Profile original and variants using immutable reference identities.

    Default cohort is the FOV intersection, reported before work starts. Explicit
    fov_ids must occur in EVERY source. CP supports process workers per FOV;
    MorphEm remains sequential. CP batch_size controls output write chunks.
    """
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (workers, threads_per_worker)):
        raise ValueError('workers and threads_per_worker must be positive integers')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    reference = _collection(reference)
    if reference.spec.get('kind') == 'degradation':
        raise ValueError('reference must be the original collection')
    sources = [reference, *map(_collection, variants)]
    if len({s.fingerprint for s in sources}) != len(sources):
        raise ValueError('Duplicate source collections')
    output = Path(output).resolve()
    for source in sources:
        if output == source.root or output in source.root.parents or source.root in output.parents:
            raise ValueError('Profile output must be separate from image collections')
    common_fovs = set.intersection(*(set(s.spec['items']) for s in sources))
    ids = sorted(common_fovs) if fov_ids is None else list(fov_ids)
    if not ids or len(ids) != len(set(ids)) or set(ids) - common_fovs:
        raise ValueError('Select nonempty, unique FOV IDs available in every source')
    if channels is None:
        channels = sorted(set.intersection(*(set(s.images.loc[s.images.role.eq('intensity'), 'channel']) for s in sources)))
    channels = list(channels)
    if not channels or len(channels) != len(set(channels)):
        raise ValueError('Select nonempty distinct channels common to all sources')
    crop_descriptor = crop_policy.descriptor()
    description = json.loads(json.dumps(profiler.describe(), allow_nan=False))
    if workers > 1 and description['name'] != 'cp_measure':
        raise ValueError('Process workers currently support only CPMeasureProfiler')
    names = description['feature_names']
    if not names or len(names) != len(set(names)) or description['input_scope'] not in ('patch', 'full_fov'):
        raise ValueError('Invalid profiler feature names/input_scope')
    if crop_policy is None:
        crop_policy = CropPolicy()
    else:
        raise NotImplementedError('Custom crop policies are not supported')
    
    feature_set_id = 'features_' + digest([description, crop_descriptor if description['input_scope'] == 'patch' else None])[:24]
    source_specs = []
    for source in sources:
        if source is not reference and source.spec.get('reference_fingerprint') != reference.fingerprint:
            raise ValueError('Variant was not generated from this exact reference')
        if source is not reference and source.spec['transform']['mask_policy'] != 'fixed_reference':
            raise ValueError('Only fixed-reference-mask variants are supported')
        for fid in ids:
            ref_rows = reference.images.loc[reference.images.fov_id.eq(fid)]
            rows = source.images.loc[source.images.fov_id.eq(fid)]
            ref_fov = reference.fovs.loc[reference.fovs.fov_id.eq(fid)].iloc[0]
            src_fov = source.fovs.loc[source.fovs.fov_id.eq(fid)].iloc[0]
            if any(ref_fov[k] != src_fov[k] for k in ('mask_set_id', 'height', 'width')):
                raise ValueError(f'{fid}: mask identity or image shape differs')
            for channel in channels:
                expected = ref_rows.loc[ref_rows.role.eq('intensity') & ref_rows.channel.eq(channel)]
                actual = rows.loc[rows.role.eq('intensity') & rows.channel.eq(channel)]
                if len(expected) != 1 or len(actual) != 1:
                    raise ValueError(f'{fid}: missing/duplicate channel {channel}')
                if actual.iloc[0].reference_image_id != expected.iloc[0].image_id:
                    raise ValueError('Image reference mapping differs')
                if actual.iloc[0].intensity_scale != 'raw' or actual.iloc[0].normalization_divisor != 65535:
                    raise ValueError('Pilot requires raw intensity units with divisor 65535')
        source_specs.append({'source_id': 'source_' + source.fingerprint[:24],
                             'dataset_id': source.spec['dataset_id'],
                             'fingerprint': source.fingerprint,
                             'variant_id': source.spec.get('variant_id', 'reference'),
                             'uri': os.path.relpath(source.root, output)})
    spec = {'version': 'profile-v1', 'reference_fingerprint': reference.fingerprint,
            'sources': source_specs, 'fov_ids': ids, 'channels': channels,
            'crop_policy': crop_descriptor, 'profiler': description,
            'feature_set_id': feature_set_id, 'batch_size': batch_size}
    plan = None
    if sampling_policy is not None and sampling_plan is not None:
        raise ValueError('Supply a sampling policy OR a saved plan, not both')
    if sampling_policy is not None or sampling_plan is not None:
        # Only masks/object tables are read here, before any expensive profiling.
        candidates = []
        for fid in ids:
            planes = reference.read_fov(fid, channels=[], include_masks=True)
            objects, relationships = reference.object_tables(fid)
            candidates.append(make_anchors(fid, objects, relationships,
                planes[('labels', 'nuclei')], planes[('labels', 'cells')], crop_policy))
        candidates = pd.concat(candidates, ignore_index=True)
        if sampling_plan is not None:
            plan = reuse_plan(candidates, sampling_plan)
            if set(plan.reference_fingerprint) != {reference.fingerprint}:
                raise ValueError('Sampling plan belongs to a different reference collection')
            policy_description = {'reused_plan': True}
        else:
            policy_description = sampling_policy.descriptor()
            metadata = reference.metadata()
            metadata = metadata.loc[metadata.fov_id.isin(ids)]
            plan = sample_anchors(candidates, metadata, sampling_policy)
            plan['reference_fingerprint'] = reference.fingerprint
        spec['sampling'] = {'policy': policy_description, 'plan_digest': digest(plan.to_json()),
                            'n_candidates': len(plan), 'n_selected': int(plan.selected.sum())}
        print(f"Selected {int(plan.selected.sum())}/{int(plan.crop_eligible.sum())} eligible nuclei", flush=True)
    run_id = 'profile_' + digest(spec)[:24]
    spec['profile_run_id'] = run_id
    print(f'Profiling cohort: {len(ids)}/{len(reference.fovs)} reference FOVs; '
          f'{len(sources)} sources; channels={channels}', flush=True)
    if dry_run:
        return spec
    start_run = time.perf_counter()
    with run_lock(output, spec):
        if plan is not None:
            plan.to_parquet(output / 'sampling_plan.parquet.tmp', index=False)
            (output / 'sampling_plan.parquet.tmp').replace(output / 'sampling_plan.parquet')
            sampling_summary(plan).to_parquet(output / 'sampling_summary.parquet', index=False)
        if description.get('feature_catalog'):
            pd.DataFrame(description['feature_catalog']).to_parquet(output / 'feature_catalog.parquet', index=False)
        for folder in ('shards', 'anchors', 'warnings'):
            (output / folder).mkdir(exist_ok=True)
        if description['name'] == 'cp_measure':
            print(f"CP workers={workers}; RuntimeWarnings -> {output / 'warnings'}/*.jsonl", flush=True)
        records = []
        anchor_paths = [output / 'anchors' / f'{fid}.parquet' for fid in ids]
        tasks = ((fid, None if plan is None else plan.loc[plan.fov_id.eq(fid)].reset_index(drop=True))
                 for fid in ids)
        if workers > 1:
            from .parallel import parallel_fovs
            results = parallel_fovs(tasks, reference, sources, output, crop_policy, spec,
                                    workers, threads_per_worker)
        else:
            results = (_profile_fov(fid, reference, sources, output, profiler, crop_policy, spec, selected)
                       for fid, selected in tasks)
        try:
            for completed in results:
                for done in completed:
                    records.append(done)
                    variant = next(x['variant_id'] for x in source_specs if x['source_id'] == done['source_id'])
                    status = 'resumed' if done['resumed'] else 'completed'
                    print(f"[{len(records)}/{len(ids) * len(sources)}] {done['fov_id']} {variant} "
                          f"{status} rows={done['n_rows']} {done['elapsed_seconds']:.2f}s", flush=True)
        finally:
            results.close()
        for source in sources:
            if (source.root / '_SUCCESS').read_text().strip() != source.fingerprint:
                raise RuntimeError('Image collection changed before profiling completion')
        merge_parquet(anchor_paths, output / 'anchors.parquet')
        rank = {(fid, src['source_id']): (i, j) for i, fid in enumerate(ids) for j, src in enumerate(source_specs)}
        records.sort(key=lambda r: rank[r['fov_id'], r['source_id']])
        manifest = pd.DataFrame(records)
        manifest.to_parquet(output / 'manifest.parquet.tmp', index=False)
        (output / 'manifest.parquet.tmp').replace(output / 'manifest.parquet')
        atomic_json(output / 'invocation.json', {'wall_seconds': time.perf_counter() - start_run,
                    'completed_shards': len(records), 'resumed_shards': int(manifest.resumed.sum()),
                    'workers': workers, 'threads_per_worker': threads_per_worker if workers > 1 else None})
        (output / '_SUCCESS.tmp').write_text(digest(spec))
        (output / '_SUCCESS.tmp').replace(output / '_SUCCESS')
    return manifest


def _profile_fov(fid, reference, sources, output, profiler, crop_policy, spec,
                 anchors_override=None, stop_event=None):
    """One task owns one FOV's anchors and all its independent source shards."""
    description = spec['profiler']
    source_specs = spec['sources']
    run_id, feature_set_id = spec['profile_run_id'], spec['feature_set_id']
    channels, batch_size = spec['channels'], spec['batch_size']
    schema = feature_schema(len(description['feature_names']))
    records = []
    # The exact same anchors, masks, and reference object table are reused.
    mask_planes = reference.read_fov(fid, channels=[], include_masks=True)
    masks = {kind: mask_planes[('labels', kind)] for kind in ('nuclei', 'cells')}
    objects, relationships = reference.object_tables(fid)
    anchors = make_anchors(fid, objects, relationships, masks['nuclei'], masks['cells'], crop_policy)
    if anchors_override is not None:
        anchors = anchors_override
    anchor_path = output / 'anchors' / f'{fid}.parquet'
    if not anchor_path.exists():
        temp = anchor_path.with_suffix('.parquet.tmp')
        anchors.to_parquet(temp, index=False)
        temp.replace(anchor_path)
    else:
        # Detect incompatible anchor edits even when restarting the same run.
        saved = pd.read_parquet(anchor_path)
        pd.testing.assert_frame_equal(saved, anchors, check_dtype=False)
    for source, source_spec in zip(sources, source_specs):
        if stop_event is not None and stop_event.is_set():
            from concurrent.futures import CancelledError
            raise CancelledError('Profiling stopped between shards')
        source_id = source_spec['source_id']
        unit_id = digest([source_id, fid])[:24]
        unit_fingerprint = digest([run_id, source.spec['items'][fid], anchors.to_json()])
        destination = output / 'shards' / unit_id
        if destination.exists():
            done = json.loads((destination / 'complete.json').read_text())
            if done['fingerprint'] != unit_fingerprint or not (destination / 'features.parquet').is_file():
                raise ValueError('Invalid completed profiling shard')
            records.append({**done, 'resumed': True})
            continue
        temp = output / 'shards' / ('.' + unit_id + '.' + uuid.uuid4().hex)
        temp.mkdir()
        start = time.perf_counter()
        try:
            before = time.perf_counter()
            planes = source.read_fov(fid, channels=channels, include_masks=False)
            read_seconds = time.perf_counter() - before
            images = {c: planes[('intensity', c)] for c in channels}
            if any(a.shape != masks['nuclei'].shape for a in images.values()):
                raise ValueError('Intensity/reference-mask shapes differ')
            image_rows = source.images.loc[source.images.fov_id.eq(fid) & source.images.role.eq('intensity') & source.images.channel.isin(channels)]
            context = FOVContext(fid, images, image_rows, masks, objects, relationships, anchors)
            common = {'profile_run_id': run_id, 'feature_set_id': feature_set_id,
                      'source_id': source_id, 'dataset_id': source.spec['dataset_id'],
                      'variant_id': source_spec['variant_id'], 'fov_id': fid,
                      'mask_set_id': reference.fovs.loc[reference.fovs.fov_id.eq(fid), 'mask_set_id'].iloc[0]}
            count, seen = 0, set()
            before = time.perf_counter()
            batches = profiler.profile_fov(context, batch_size)
            if description['name'] == 'cp_measure':
                batches = logged_cp_batches(batches, output / 'warnings' / f'{unit_id}.jsonl', common)
            with pq.ParquetWriter(temp / 'features.parquet', schema, compression='zstd') as writer:
                for batch in batches:
                    table = feature_table(batch.observations, batch.values, context,
                                          common=common, schema=schema, description=description)
                    for obs in batch.observations.to_dict('records'):
                        key = (obs['object_id'], obs['crop_id'], tuple(obs['channels']))
                        if key in seen:
                            raise ValueError('Duplicate feature observation')
                        seen.add(key)
                    writer.write_table(table)
                    count += len(table)
            if description['name'] == 'morphem':
                expected = {(r.object_id, r.crop_id, (channel,))
                            for r in anchors.loc[anchors.eligible.eq(True)].itertuples(index=False)
                            for channel in channels}
                if seen != expected:
                    raise ValueError('MorphEm output did not cover every eligible nucleus/channel')
            profile_seconds = time.perf_counter() - before
            for s in (reference, source):
                if (s.root / '_SUCCESS').read_text().strip() != s.fingerprint:
                    raise RuntimeError('Image collection changed during profiling')
            done = {'fingerprint': unit_fingerprint, 'source_id': source_id, 'fov_id': fid,
                    'feature_uri': f'shards/{unit_id}/features.parquet', 'n_rows': count,
                    'n_nuclei': len(anchors), 'n_eligible': int(anchors.eligible.sum()),
                    'read_seconds': read_seconds, 'profile_write_seconds': profile_seconds,
                    'elapsed_seconds': time.perf_counter() - start, 'resumed': False, 'worker_pid': os.getpid()}
            atomic_json(temp / 'complete.json', done)
            temp.rename(destination)
            records.append(done)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return records
